from gym.spaces import Box
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from marl.models.base.base_rnn import Base_RNN
import copy
from ray.rllib.utils.annotations import override
from functools import reduce

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
            if isinstance(self.custom_config["space_act"], Box):  # continues
                input_size = cc_input_dim + num_outputs * (self.custom_config["num_agents"] - 1) // 2
            else:
                input_size = cc_input_dim + num_outputs * (self.custom_config["num_agents"] - 1)
        else:
            input_size = cc_input_dim

        self.central_vf = nn.Sequential(
            nn.Linear(input_size, 1),
        )

        if self.custom_config["algorithm"] in ["coma"]:
            self.q_flag = True
            self.value_branch = nn.Linear(self.input_dim, num_outputs)
            self.central_vf = nn.Sequential(
                nn.Linear(input_size, num_outputs),
            )

    def central_value_function(self, state, opponent_actions=None):
        B = state.shape[0]

        if "conv_layer" in self.custom_config["model_arch_args"]:
            x = state.reshape(-1, self.state_dim[0], self.state_dim[1], self.state_dim[2]).permute(0, 3, 1, 2)
            x = self.cc_encoder(x)
            x = torch.mean(x, (2, 3))
        else:
            x = self.cc_encoder(state)

        if opponent_actions is not None:
            if isinstance(self.custom_config["space_act"], Box):  # continues
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