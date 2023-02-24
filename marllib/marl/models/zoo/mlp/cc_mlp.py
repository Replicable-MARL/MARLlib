import numpy as np
from gym.spaces import Box
from functools import reduce
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import copy
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
from marllib.marl.models.zoo.mlp.base_mlp import Base_MLP

torch, nn = try_import_torch()


class CC_MLP(Base_MLP):

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
        cc_layers = []
        if "state" not in self.full_obs_space.spaces:
            self.state_dim = self.full_obs_space["obs"].shape
            cc_input_dim = self.encoder_layer_dim[-1] * self.custom_config["num_agents"]

            self.cc_value_encoder = copy.deepcopy(self.value_encoder)
        else:
            self.state_dim = self.full_obs_space["state"].shape
            cc_input_dim = self.state_dim[0] + self.obs_size

            for cc_out_dim in self.encoder_layer_dim:
                cc_layers.append(
                    SlimFC(in_size=cc_input_dim,
                           out_size=cc_out_dim,
                           initializer=normc_initializer(1.0),
                           activation_fn=self.activation))
                cc_input_dim = cc_out_dim

            self.cc_value_encoder = nn.Sequential(*copy.deepcopy(cc_layers))

        if self.custom_config["opp_action_in_cc"]:
            if isinstance(self.custom_config["space_act"], Box):  # continuous
                cc_input_dim = cc_input_dim + num_outputs * (self.custom_config["num_agents"] - 1) // 2
            else:
                cc_input_dim = cc_input_dim + num_outputs * (self.custom_config["num_agents"] - 1)

        self.cc_value_branch = SlimFC(
            in_size=cc_input_dim,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        self.q_flag = False
        if self.custom_config["algorithm"] in ["coma"]:
            self.q_flag = True
            self.cc_value_branch = SlimFC(
                in_size=cc_input_dim,
                out_size=num_outputs,
                initializer=normc_initializer(0.01),
                activation_fn=None)

    def central_value_function(self, state, opponent_actions=None) -> TensorType:
        assert self._features is not None, "must call forward() first"
        B = state.shape[0]

        if "conv_layer" in self.custom_config["model_arch_args"]:
            x = state.reshape(-1, self.state_dim[0], self.state_dim[1], self.state_dim_last).permute(0, 3, 1, 2)
            x = self.cc_value_encoder(x)
            x = torch.mean(x, (2, 3))
        else:
            x = self.cc_value_encoder(state)

        if opponent_actions is None:
            x = torch.cat([x.reshape(B, -1)], 1)
        else:
            if isinstance(self.custom_config["space_act"], Box):  # continuous
                opponent_actions_ls = [opponent_actions[:, i, :]
                                       for i in
                                       range(self.n_agents - 1)]
            else:
                opponent_actions_ls = [
                    torch.nn.functional.one_hot(opponent_actions[:, i].long(), self.num_outputs).float()
                    for i in
                    range(self.n_agents - 1)]

            x = torch.cat([x.reshape(B, -1)] + opponent_actions_ls, 1)

        if self.q_flag:
            return torch.reshape(self.cc_value_branch(x), [-1, self.num_outputs])
        else:
            return torch.reshape(self.cc_value_branch(x), [-1])

    @override(Base_MLP)
    def critic_parameters(self):
        critics = [
            self.cc_value_encoder,
            self.cc_value_branch,
        ]
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), critics))

    def link_other_agent_policy(self, agent_id, policy):
        if agent_id in self.other_policies:
            if self.other_policies[agent_id] != policy:
                raise ValueError('the policy is not same with the two time look up')
        else:
            self.other_policies[agent_id] = policy

    def set_train_batch(self, batch):
        self._train_batch_ = batch.copy()

        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                try:
                    self._train_batch_[key] = torch.Tensor(value)
                except TypeError as e:
                    # print(f'key: {key} cannot be convert to Tensor')
                    pass

    def get_train_batch(self):
        return self._train_batch_

    def get_actions(self):
        return self(self._train_batch_)

    def update_actor(self, loss, lr, grad_clip):
        CC_MLP.update_use_torch_adam(
            loss=(-1 * loss),
            optimizer=self.actor_optimizer,
            parameters=self.parameters(),
            grad_clip=grad_clip
        )

    def update_critic(self, loss, lr, grad_clip):
        CC_MLP.update_use_torch_adam(
            loss=loss,
            optimizer=self.critic_optimizer,
            parameters=self.critic_parameters(),
            grad_clip=grad_clip
        )

    @staticmethod
    def update_use_torch_adam(loss, parameters, optimizer, grad_clip):
        optimizer.zero_grad()
        loss.backward()
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in parameters if p.grad is not None]))
        torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        optimizer.step()

    def __update_adam(self, loss, parameters, adam_info, lr, grad_clip, step, maximum=False):

        for p in self.parameters():
            p.grad = None

        gradients = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
        total_norm = torch.norm(torch.stack([torch.norm(grad) for grad in gradients]))
        max_norm = grad_clip
        clip_coef = max_norm / (total_norm + 1e-6)

        if clip_coef < 1:
            for g in gradients:
                g.detach().mul_(clip_coef.to(g.device))

        after_total_norm = torch.norm(torch.stack([torch.norm(grad) for grad in gradients]))
        if total_norm != after_total_norm:
            print(f'before clip norm: {total_norm}')
            print(f'after clip norm: {after_total_norm}')
        if after_total_norm - grad_clip > 1:
            raise ValueError(f'grad clip error!, after clip norm: {after_total_norm}, clip norm threshold: {grad_clip}')

        beta1, beta2 = 0.9, 0.999
        eps = 1e-05

        real_gradients = []

        if maximum:
            # for i, param in enumerate(parameters):
            for grad in gradients:
                # gradients[i] = -gradients[i]  # get maximize
                grad = -1 * grad
                real_gradients.append(grad)

            gradients = real_gradients

        m_v = []
        v_v = []

        if len(adam_info['m']) == 0:
            adam_info['m'] = [0] * len(gradients)
            adam_info['v'] = [0] * len(gradients)

        for i, g in enumerate(gradients):
            mt = beta1 * adam_info['m'][i] + (1 - beta1) * g
            vt = beta2 * adam_info['v'][i] + (1 - beta2) * (g ** 2)

            m_t_bar = mt / (1 - beta1 ** step)
            v_t_bar = vt / (1 - beta2 ** step)

            vector_to_parameters(
                parameters_to_vector([parameters[i]]) - parameters_to_vector(
                    lr * m_t_bar / (torch.sqrt(v_t_bar) + eps)),
                [parameters[i]],
            )

            m_v.append(mt)
            v_v.append(vt)

        step += 1

        adam_info['m'] = m_v
        adam_info['v'] = v_v

        return step
