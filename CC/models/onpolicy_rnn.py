from ray.rllib.utils.torch_ops import FLOAT_MIN
import numpy as np
from typing import Dict, List, Any, Union
from gym.spaces import Box
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch, \
    TensorType
from ray.rllib.policy.rnn_sequencing import add_time_dimension
import copy

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class Onpolicy_Universal_Model(TorchRNN, nn.Module):

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # judge the model arch
        custom_config = model_config["custom_model_config"]
        full_obs_space = getattr(obs_space, "original_space", obs_space)

        # encoder
        layers = []
        if "fc_layer" in custom_config["model_arch_args"]:
            self.obs_size = full_obs_space['obs'].shape[0]
            input_dim = self.obs_size
            for i in range(custom_config["model_arch_args"]["fc_layer"]):
                out_dim = custom_config["model_arch_args"]["out_dim_fc_{}".format(i)]
                fc_layer = nn.Linear(input_dim, out_dim)
                layers.append(fc_layer)
                input_dim = out_dim
        elif "conv_layer" in custom_config["model_arch_args"]:
            self.obs_size = full_obs_space['obs'].shape
            input_dim = self.obs_size[2]
            for i in range(custom_config["model_arch_args"]["conv_layer"]):
                conv_f = nn.Conv2d(
                    in_channels=input_dim,
                    out_channels=custom_config["model_arch_args"]["out_channel_layer_{}".format(i)],
                    kernel_size=custom_config["model_arch_args"]["kernel_size_layer_{}".format(i)],
                    stride=custom_config["model_arch_args"]["stride_layer_{}".format(i)],
                    padding=custom_config["model_arch_args"]["padding_layer_{}".format(i)],
                )
                relu_f = nn.ReLU()
                pool_f = nn.MaxPool2d(kernel_size=custom_config["model_arch_args"]["pool_size_layer_{}".format(i)])

                layers.append(conv_f)
                layers.append(relu_f)
                layers.append(pool_f)

                input_dim = custom_config["model_arch_args"]["out_channel_layer_{}".format(i)]

        else:
            raise ValueError()

        self.encoder = nn.Sequential(
            *layers
        )

        # encoder for centralized function
        if "state" not in full_obs_space.spaces:
            self.state_dim = full_obs_space["obs"].shape
            self.cc_encoder = copy.deepcopy(self.encoder)
            cc_input_dim = input_dim * custom_config["num_agents"]
        else:
            self.state_dim = full_obs_space["state"].shape
            if len(self.state_dim) > 1:  # env return a 3D global state
                cc_layers = []
                cc_input_dim = self.state_dim[2]
                for i in range(custom_config["model_arch_args"]["conv_layer"]):
                    cc_conv_f = nn.Conv2d(
                        in_channels=cc_input_dim,
                        out_channels=custom_config["model_arch_args"]["out_channel_layer_{}".format(i)],
                        kernel_size=custom_config["model_arch_args"]["kernel_size_layer_{}".format(i)],
                        stride=custom_config["model_arch_args"]["stride_layer_{}".format(i)],
                        padding=custom_config["model_arch_args"]["padding_layer_{}".format(i)],
                    )
                    cc_relu_f = nn.ReLU()
                    cc_pool_f = nn.MaxPool2d(
                        kernel_size=custom_config["model_arch_args"]["pool_size_layer_{}".format(i)])

                    cc_layers.append(cc_conv_f)
                    cc_layers.append(cc_relu_f)
                    cc_layers.append(cc_pool_f)

                    cc_input_dim = custom_config["model_arch_args"]["out_channel_layer_{}".format(i)]

                self.cc_encoder = nn.Sequential(
                    *cc_layers
                )
            else:
                cc_layers = []
                cc_input_dim = full_obs_space["state"].shape[0]
                for i in range(custom_config["model_arch_args"]["fc_layer"]):
                    cc_out_dim = custom_config["model_arch_args"]["out_dim_fc_{}".format(i)]
                    cc_fc_layer = nn.Linear(cc_input_dim, cc_out_dim)
                    cc_layers.append(cc_fc_layer)
                    cc_input_dim = cc_out_dim

                self.cc_encoder = nn.Sequential(
                    *cc_layers
                )

        # core rnn
        self.hidden_state_size = custom_config["model_arch_args"]["hidden_state_size"]

        if custom_config["model_arch_args"]["core_arch"] == "gru":
            self.rnn = nn.GRU(input_dim, self.hidden_state_size, batch_first=True)
        elif custom_config["model_arch_args"]["core_arch"] == "lstm":
            self.rnn = nn.LSTM(input_dim, self.hidden_state_size, batch_first=True)
        else:
            raise ValueError()
        # action branch and value branch
        self.action_branch = nn.Linear(self.hidden_state_size, num_outputs)
        self.value_branch = nn.Linear(self.hidden_state_size, 1)

        # Holds the current "base" output (before logits layer).
        self._features = None

        # Central VF
        if custom_config["opp_action_in_cc"]:
            if isinstance(custom_config["space_act"], Box):  # continues
                input_size = cc_input_dim + 2 * (custom_config["num_agents"] - 1)
            else:
                input_size = cc_input_dim + num_outputs * (custom_config["num_agents"] - 1)
        else:
            input_size = cc_input_dim
        self.central_vf = nn.Sequential(
            nn.Linear(input_size, 1),
        )

        self.coma_flag = False
        if "coma" == custom_config["algorithm"]:
            self.coma_flag = True
            self.value_branch = nn.Linear(self.hidden_state_size, num_outputs)
            self.central_vf = nn.Sequential(
                nn.Linear(input_size, num_outputs),
            )

        # record the custom config
        self.custom_config = custom_config
        self.n_agents = custom_config["num_agents"]

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        if self.custom_config["model_arch_args"]["core_arch"] == "gru":
            h = [
                self.value_branch.weight.new(1, self.hidden_state_size).zero_().squeeze(0),
            ]
        else:  # lstm
            h = [
                self.value_branch.weight.new(1, self.hidden_state_size).zero_().squeeze(0),
                self.value_branch.weight.new(1, self.hidden_state_size).zero_().squeeze(0)
            ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(ModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """
        Adds time dimension to batch before sending inputs to forward_rnn()
        """
        if self.custom_config["global_state_flag"] or self.custom_config["mask_flag"]:
            flat_inputs = input_dict["obs"]["obs"].float()
            # Convert action_mask into a [0.0 || -inf]-type mask.
            if self.custom_config["mask_flag"]:
                action_mask = input_dict["obs"]["action_mask"]
                inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        else:
            flat_inputs = input_dict["obs"]["obs"].float()

        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        # if seq_lens.shape[0] == 0:
        #     print(1)
        max_seq_len = flat_inputs.shape[0] // seq_lens.shape[0]

        self.time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            flat_inputs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )
        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])

        if self.custom_config["mask_flag"]:
            output = output + inf_mask

        return output, new_state

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        # Extract the available actions tensor from the observation.

        # Compute the unmasked logits.
        if "conv_layer" in self.custom_config["model_arch_args"]:
            x = inputs.reshape(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4]).permute(0, 3, 1, 2)
            x = self.encoder(x)
            x = torch.mean(x, (2, 3))
            x = x.reshape(inputs.shape[0], inputs.shape[1], -1)
        else:
            x = self.encoder(inputs)

        x = nn.functional.relu(x)

        if self.custom_config["model_arch_args"]["core_arch"] == "gru":
            self._features, h = self.rnn(x, torch.unsqueeze(state[0], 0))
            logits = self.action_branch(self._features)
            return logits, [torch.squeeze(h, 0)]

        elif self.custom_config["model_arch_args"]["core_arch"] == "lstm":
            self._features, [h, c] = self.rnn(
                x, [torch.unsqueeze(state[0], 0),
                    torch.unsqueeze(state[1], 0)])
            logits = self.action_branch(self._features)
            return logits, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

        else:
            raise ValueError()

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
        if self.coma_flag:
            return torch.reshape(self.central_vf(x), [-1, self.num_outputs])
        else:
            return torch.reshape(self.central_vf(x), [-1])
