import numpy as np
from gym.spaces import Box
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from marllib.marl.models.base.base_rnn import Base_RNN
import copy
from ray.rllib.utils.annotations import override
from functools import reduce
from icecream import ic
from torch.optim import RMSprop, Adam
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from marllib.marl.algos.utils.distributions import init
from marllib.marl.algos.utils.mlp import MLPBase

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class CC_RNN(Base_RNN):

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

        # extra encoder for centralized VF
        input_dim = self.input_dim
        if "state" not in self.full_obs_space.spaces:
            self.state_dim = self.full_obs_space["obs"].shape
            self.cc_encoder = copy.deepcopy(self.encoder)
            cc_input_dim = input_dim * self.custom_config["num_agents"]
        else:
            self.state_dim = self.full_obs_space["state"].shape
            if len(self.state_dim) > 1:  # env return a 3D global state
                cc_layers = []
                self.state_dim_last = self.full_obs_space["state"].shape[2] + self.full_obs_space["obs"].shape[2]
                cc_input_dim = self.state_dim_last
                for i in range(self.custom_config["model_arch_args"]["conv_layer"]):
                    cc_conv_f = nn.Conv2d(
                        in_channels=cc_input_dim,
                        out_channels=self.custom_config["model_arch_args"]["out_channel_layer_{}".format(i)],
                        kernel_size=self.custom_config["model_arch_args"]["kernel_size_layer_{}".format(i)],
                        stride=self.custom_config["model_arch_args"]["stride_layer_{}".format(i)],
                        padding=self.custom_config["model_arch_args"]["padding_layer_{}".format(i)],
                    )
                    cc_relu_f = nn.ReLU()
                    cc_pool_f = nn.MaxPool2d(
                        kernel_size=self.custom_config["model_arch_args"]["pool_size_layer_{}".format(i)])

                    cc_layers.append(cc_conv_f)
                    cc_layers.append(cc_relu_f)
                    cc_layers.append(cc_pool_f)

                    cc_input_dim = self.custom_config["model_arch_args"]["out_channel_layer_{}".format(i)]

            else:
                cc_layers = []
                cc_input_dim = self.full_obs_space["state"].shape[0] + self.full_obs_space["obs"].shape[0]
                for i in range(self.custom_config["model_arch_args"]["fc_layer"]):
                    cc_out_dim = self.custom_config["model_arch_args"]["out_dim_fc_{}".format(i)]
                    cc_fc_layer = nn.Linear(cc_input_dim, cc_out_dim)
                    cc_layers.append(cc_fc_layer)
                    cc_input_dim = cc_out_dim

            cc_layers.append(nn.Tanh())
            self.cc_encoder = nn.Sequential(
                *cc_layers
            )

        # Central VF
        if self.custom_config["opp_action_in_cc"]:
            if isinstance(self.custom_config["space_act"], Box):  # continuous
                input_size = cc_input_dim + num_outputs * (self.custom_config["num_agents"] - 1) // 2
            else:
                input_size = cc_input_dim + num_outputs * (self.custom_config["num_agents"] - 1)
        else:
            input_size = cc_input_dim

        self.central_vf = nn.Sequential(
            nn.Linear(input_size, 1),
        )

        if self.custom_config['algorithm'].lower() in ['happo']:
            # set actor
            def init_(m, value):
                return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), value)

            # self.encoder = MLPBase(
            #     obs_space.shape
            # )

            self.action_branch = init_(nn.Linear(self.hidden_state_size, num_outputs), value=self.custom_config['gain'])

            self.central_vf = nn.Sequential(
                init_(nn.Linear(input_size, 1), value=1)
            )

            self.actors = [self.encoder, self.rnn, self.action_branch]

            # self.actor_optimizer = Adam(params=self.actor_parameters(), lr=self.custom_config['actor_lr'])
            self.actor_optimizer = Adam(params=self.parameters(), lr=self.custom_config['actor_lr'])
            # self.critic_optimizer = Adam(params=self.critic_parameters(), lr=self.custom_config['critic_lr'])

        if self.custom_config["algorithm"] in ["coma"]:
            self.q_flag = True
            self.value_branch = nn.Linear(self.input_dim, num_outputs)
            self.central_vf = nn.Sequential(
                nn.Linear(input_size, num_outputs),
            )

        self.other_policies = {}

        self.actor_adam_update_info = {'m': [], 'v': []}
        self.critic_adam_update_info = {'m': [], 'v': []}
        self.__t_actor = 1
        self.__t_critic = 1

        self._train_batch_ = None

    def central_value_function(self, state, opponent_actions=None):
        B = state.shape[0]

        if "conv_layer" in self.custom_config["model_arch_args"]:
            x = state.reshape(-1, self.state_dim[0], self.state_dim[1], self.state_dim_last).permute(0, 3, 1, 2)
            x = self.cc_encoder(x)
            x = torch.mean(x, (2, 3))
        else:
            x = self.cc_encoder(state)

        if opponent_actions is not None:
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

        else:
            x = torch.cat([x.reshape(B, -1)], 1)

        if self.q_flag:
            return torch.reshape(self.central_vf(x), [-1, self.num_outputs])
        else:
            return torch.reshape(self.central_vf(x), [-1])

    @override(Base_RNN)
    def critic_parameters(self):
        critics = [
            self.cc_encoder,
            self.central_vf,
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
        # self.__t_actor = self.__update_adam(
        #     loss=loss, parameters=self.actor_parameters(),
        #     adam_info=self.actor_adam_update_info,
        #     lr=lr, grad_clip=grad_clip, step=self.__t_actor,
        #     maximum=maximum,
        # )
        CC_RNN.update_use_torch_adam(
            loss=(-1 * loss),
            optimizer=self.actor_optimizer,
            parameters=self.parameters(),
            grad_clip=grad_clip
        )

    def update_critic(self, loss, lr, grad_clip):
        # self.__t_critic = self.__update_adam(
        #     loss=loss, parameters=self.critic_parameters(),
        #     adam_info=self.critic_adam_update_info,
        #     lr=lr, grad_clip=grad_clip, step=self.__t_critic,
        # )
        CC_RNN.update_use_torch_adam(
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
        # after_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        # after_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in parameters if p.grad is not None]))

        # if before_norm != after_norm:
        #     print(f'before clip norm: {before_norm}')
        #     print(f'after clip norm: {after_norm}')
        # if after_norm - grad_clip > 1:
        #     raise ValueError(f'grad clip error!, after clip norm: {after_norm}, clip norm threshold: {grad_clip}')

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
                parameters_to_vector([parameters[i]]) - parameters_to_vector(lr * m_t_bar / (torch.sqrt(v_t_bar) + eps)),
                [parameters[i]],
            )

            m_v.append(mt)
            v_v.append(vt)

        step += 1

        adam_info['m'] = m_v
        adam_info['v'] = v_v

        return step

        # m = beta1 * self.adam_update_info['m'] + (1 - beta1) * gradients
        # v = beta2 * self.adam_update_info['v'] + (1 - beta2) * (gradients**2)

        # self.adam_update_info['m'] = m
        # self.adam_update_info['v'] = v
