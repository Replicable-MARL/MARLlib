# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from gym.spaces import Box, MultiDiscrete
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from marllib.marl.models.zoo.rnn.base_rnn import BaseRNN
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from functools import reduce
from torch.optim import Adam
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from marllib.marl.models.zoo.encoder.cc_encoder import CentralizedEncoder

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class CentralizedCriticRNN(BaseRNN):

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name, **kwargs)

        # encoder for centralized VF
        self.cc_vf_encoder = CentralizedEncoder(model_config, self.full_obs_space)

        # Central VF
        if self.custom_config["opp_action_in_cc"]:
            if isinstance(self.custom_config["space_act"], Box):  # continuous
                input_size = self.cc_vf_encoder.output_dim + num_outputs * (self.n_agents - 1) // 2
            else:
                input_size = self.cc_vf_encoder.output_dim + num_outputs * (self.n_agents - 1)
        else:
            input_size = self.cc_vf_encoder.output_dim

        self.cc_vf_branch = SlimFC(
            in_size=input_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        if self.custom_config['algorithm'].lower() in ['happo']:
            # set actor
            def init_(m, value):
                def init(module, weight_init, bias_init, gain=1):
                    weight_init(module.weight.data, gain=gain)
                    bias_init(module.bias.data)
                    return module

                return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), value)

            self.p_branch = init_(nn.Linear(self.hidden_state_size, num_outputs), value=self.custom_config['gain'])

            self.cc_vf_branch = nn.Sequential(
                init_(nn.Linear(input_size, 1), value=1)
            )

            self.actors = [self.p_encoder, self.rnn, self.p_branch]

            # self.actor_optimizer = Adam(params=self.actor_parameters(), lr=self.custom_config['actor_lr'])
            self.actor_optimizer = Adam(params=self.parameters(), lr=self.custom_config['actor_lr'])
            # self.critic_optimizer = Adam(params=self.critic_parameters(), lr=self.custom_config['critic_lr'])

        if self.custom_config["algorithm"] in ["coma"]:
            self.q_flag = True
            self.cc_vf_branch = SlimFC(
                in_size=input_size,
                out_size=num_outputs,
                initializer=normc_initializer(0.01),
                activation_fn=None)

        self.other_policies = {}

        self.actor_adam_update_info = {'m': [], 'v': []}
        self.critic_adam_update_info = {'m': [], 'v': []}
        self.__t_actor = 1
        self.__t_critic = 1

        self._train_batch_ = None

    def central_value_function(self, state, opponent_actions=None):
        assert self._features is not None, "must call forward() first"
        B = state.shape[0]
        x = self.cc_vf_encoder(state)
        if opponent_actions is not None:
            if isinstance(self.custom_config["space_act"], Box):  # continuous
                opponent_actions_ls = [opponent_actions[:, i, :]
                                       for i in
                                       range(self.n_agents - 1)]
            elif isinstance(self.custom_config["space_act"], MultiDiscrete):
                opponent_actions_ls = []
                action_space_ls = [single_action_space.n for single_action_space in self.action_space]
                for i in range(self.n_agents - 1):
                    opponent_action_ls = []
                    for single_action_index, single_action_space in enumerate(action_space_ls):
                        opponent_action = torch.nn.functional.one_hot(
                            opponent_actions[:, i, single_action_index].long(), single_action_space).float()
                        opponent_action_ls.append(opponent_action)
                    opponent_actions_ls.append(torch.cat(opponent_action_ls, axis=1))

            else:
                opponent_actions_ls = [
                    torch.nn.functional.one_hot(opponent_actions[:, i].long(), self.num_outputs).float()
                    for i in
                    range(self.n_agents - 1)]
            x = torch.cat([x.reshape(B, -1)] + opponent_actions_ls, 1)
        else:
            x = torch.cat([x.reshape(B, -1)], 1)

        if self.q_flag:
            return torch.reshape(self.cc_vf_branch(x), [-1, self.num_outputs])
        else:
            return torch.reshape(self.cc_vf_branch(x), [-1])

    @override(BaseRNN)
    def critic_parameters(self):
        critics = [self.cc_vf_encoder, self.cc_vf_branch, ]
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), critics))

    def link_other_agent_policy(self, agent_id, policy):
        if agent_id in self.other_policies:
            if self.other_policies[agent_id] != policy:
                raise ValueError('the policy is not same with the two time look up')
        else:
            self.other_policies[agent_id] = policy

    def update_actor(self, loss, lr, grad_clip):
        CentralizedCriticRNN.update_use_torch_adam(
            loss=(-1 * loss),
            optimizer=self.actor_optimizer,
            parameters=self.parameters(),
            grad_clip=grad_clip
        )

    @staticmethod
    def update_use_torch_adam(loss, parameters, optimizer, grad_clip):
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        optimizer.step()
