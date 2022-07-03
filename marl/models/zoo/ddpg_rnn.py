from ray.rllib.utils.torch_ops import FLOAT_MIN
import numpy as np
from typing import Dict, List, Any, Union
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch, \
    TensorType
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from marl.models.zoo.mixers import QMixer, VDNMixer

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class DDPG_RNN(TorchRNN, nn.Module):
    """
    DDOG/MADDPG/FACMAC agent arch in one model
    """

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
        self.custom_config = model_config["custom_model_config"]
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)

        # encoder
        layers = []
        if "fc_layer" in self.custom_config["model_arch_args"]:
            self.obs_size = self.full_obs_space['obs'].shape[0]
            input_dim = self.obs_size
            for i in range(self.custom_config["model_arch_args"]["fc_layer"]):
                out_dim = self.custom_config["model_arch_args"]["out_dim_fc_{}".format(i)]
                fc_layer = nn.Linear(input_dim, out_dim)
                layers.append(fc_layer)
                input_dim = out_dim
        elif "conv_layer" in self.custom_config["model_arch_args"]:
            self.obs_size = self.full_obs_space['obs'].shape
            input_dim = self.obs_size[2]
            for i in range(self.custom_config["model_arch_args"]["conv_layer"]):
                conv_f = nn.Conv2d(
                    in_channels=input_dim,
                    out_channels=self.custom_config["model_arch_args"]["out_channel_layer_{}".format(i)],
                    kernel_size=self.custom_config["model_arch_args"]["kernel_size_layer_{}".format(i)],
                    stride=self.custom_config["model_arch_args"]["stride_layer_{}".format(i)],
                    padding=self.custom_config["model_arch_args"]["padding_layer_{}".format(i)],
                )
                relu_f = nn.ReLU()
                pool_f = nn.MaxPool2d(kernel_size=self.custom_config["model_arch_args"]["pool_size_layer_{}".format(i)])

                layers.append(conv_f)
                layers.append(relu_f)
                layers.append(pool_f)

                input_dim = self.custom_config["model_arch_args"]["out_channel_layer_{}".format(i)]

        else:
            raise ValueError()

        self.encoder = nn.Sequential(
            *layers
        )

        if self.custom_config["algorithm"] in ["maddpg"]:
            all_action_dim = self.custom_config["space_act"].shape[0] * self.custom_config["num_agents"]
            if self.custom_config["global_state_flag"]:
                self.state_encoder = nn.Linear(self.full_obs_space["state"].shape[0],
                                               input_dim)
            if "q" in name:
                if self.custom_config["global_state_flag"]:
                    input_dim = input_dim * 2 + all_action_dim
                else:
                    input_dim = input_dim * self.custom_config["num_agents"] + all_action_dim

        else:  # no centralized critic -> iddpg
            if "q" in name:
                input_dim = input_dim + self.custom_config["space_act"].shape[0]

        # core rnn
        self.hidden_state_size = self.custom_config["model_arch_args"]["hidden_state_size"]
        self.input_dim = input_dim

        if self.custom_config["model_arch_args"]["core_arch"] == "gru":
            self.rnn = nn.GRU(input_dim, self.hidden_state_size, batch_first=True)
        elif self.custom_config["model_arch_args"]["core_arch"] == "lstm":
            self.rnn = nn.LSTM(input_dim, self.hidden_state_size, batch_first=True)
        else:
            raise ValueError()
        # action branch and value branch
        self.action_branch = nn.Linear(self.hidden_state_size, num_outputs)
        self.value_branch = nn.Linear(self.hidden_state_size, 1)

        if self.custom_config["algorithm"] in ["facmac"]:
            # mixer:
            if self.custom_config["global_state_flag"]:
                state_dim = self.custom_config["space_obs"]["state"].shape
            else:
                state_dim = self.custom_config["space_obs"]["obs"].shape + (self.custom_config["num_agents"],)
            if self.custom_config["algo_args"]["mixer"] == "qmix":
                self.mixer = QMixer(self.custom_config, state_dim)
            elif self.custom_config["algo_args"]["mixer"] == "vdn":
                self.mixer = VDNMixer()
            else:
                raise ValueError("Unknown mixer type {}".format(self.custom_config["algo_args"]["mixer"]))

        # Holds the current "base" output (before logits layer).
        self._features = None

        # record the custom config
        self.n_agents = self.custom_config["num_agents"]
        self.q_flag = False

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
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """
        Adds time dimension to batch before sending inputs to forward_rnn()
        """

        obs_inputs = input_dict["obs"]["obs"].float()

        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        max_seq_len = obs_inputs.shape[0] // seq_lens.shape[0]

        self.time_major = self.model_config.get("_time_major", False)

        obs_inputs = add_time_dimension(
            obs_inputs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )

        if "state" in input_dict:
            state_inputs = input_dict["state"].float()
            state_inputs = add_time_dimension(
                state_inputs,
                max_seq_len=max_seq_len,
                framework="torch",
                time_major=self.time_major,
            )
        else:
            state_inputs = None

        if "opponent_actions" in input_dict:
            opp_action_inputs = input_dict["opponent_actions"].float()
            opp_action_inputs = add_time_dimension(
                opp_action_inputs,
                max_seq_len=max_seq_len,
                framework="torch",
                time_major=self.time_major,
            )
        else:
            opp_action_inputs = None

        if "actions" in input_dict:
            action_inputs = input_dict["actions"].float()
            action_inputs = add_time_dimension(
                action_inputs,
                max_seq_len=max_seq_len,
                framework="torch",
                time_major=self.time_major,
            )
        else:
            action_inputs = None

        output, new_state = self.forward_rnn(obs_inputs, action_inputs, state_inputs, opp_action_inputs, state,
                                             seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])


        return output, new_state

    @override(TorchRNN)
    def forward_rnn(self, obs_inputs, action_inputs, state_inputs, opp_action_inputs, state, seq_lens):
        # Extract the available actions tensor from the observation.
        # Compute the unmasked logits.
        if self.custom_config["algorithm"] in ["maddpg"]:
            # currently we do not support CNN in MADDPG
            if action_inputs is not None:
                B = obs_inputs.shape[0]
                L = obs_inputs.shape[1]
                obs_x = self.encoder(obs_inputs)
                if self.custom_config["global_state_flag"]:
                    state_x = self.state_encoder(state_inputs)
                    x = torch.cat((obs_x, state_x, action_inputs, opp_action_inputs.reshape(B, L, -1)), -1)
                else:
                    state_x_ls = []
                    for i in range(self.n_agents):
                        state_x = self.encoder(state_inputs[:, :, i])
                        state_x_ls.append(state_x)
                    state_x = torch.cat(state_x_ls, -1)
                    x = torch.cat((state_x, action_inputs, opp_action_inputs.reshape(B, L, -1)), -1)

            else:
                x = self.encoder(obs_inputs)

        else:
            if "conv_layer" in self.custom_config["model_arch_args"]:
                x = obs_inputs.reshape(-1, obs_inputs.shape[2], obs_inputs.shape[3], obs_inputs.shape[4]).permute(0, 3,
                                                                                                                  1,
                                                                                                                  2)
                x = self.encoder(x)
                x = torch.mean(x, (2, 3))
                x = x.reshape(obs_inputs.shape[0], obs_inputs.shape[1], -1)
            else:
                x = self.encoder(obs_inputs)

            if action_inputs is not None:
                x = torch.cat((x, action_inputs), -1)

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
            raise ValueError("rnn core_arch wrong: {}".format(self.custom_config["model_arch_args"]["core_arch"]))

    def mixing_value(self, all_agents_q, state):
        # compatiable with rllib qmix mixer
        all_agents_q = all_agents_q.view(-1, 1, self.n_agents)
        q_tot = self.mixer(all_agents_q, state)

        # shape to [B]
        return q_tot.flatten(start_dim=0)
