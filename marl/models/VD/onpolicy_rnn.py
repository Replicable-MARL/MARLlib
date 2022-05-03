from ray.rllib.utils.torch_ops import FLOAT_MIN
import numpy as np
from typing import Dict, List, Any, Union
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch, \
    TensorType
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from marl.models.VD.mixers import QMixer, VDNMixer

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

        # mixer:
        if custom_config["global_state_flag"]:
            state_dim = custom_config["space_obs"]["state"].shape
        else:
            state_dim = custom_config["space_obs"]["obs"].shape + (custom_config["num_agents"], )
        if custom_config["algo_args"]["mixer"] == "qmix":
            self.mixer = QMixer(custom_config, state_dim)
        elif custom_config["algo_args"]["mixer"] == "vdn":
            self.mixer = VDNMixer()
        else:
            raise ValueError("Unknown mixer type {}".format(custom_config["algo_args"]["mixer"]))

        # Holds the current "base" output (before logits layer).
        self._features = None

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

    def mixing_value(self, all_agents_vf, state):
        # compatiable with rllib qmix mixer
        all_agents_vf = all_agents_vf.view(-1, 1, self.n_agents)
        v_tot = self.mixer(all_agents_vf, state)

        # shape to [B]
        return v_tot.flatten(start_dim=0)
