from ray.rllib.utils.torch_ops import FLOAT_MIN
import numpy as np
from typing import Dict, List, Any, Union
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch, \
    TensorType
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from functools import reduce

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class Base_RNN(TorchRNN, nn.Module):

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

        self.input_dim = input_dim

        # obs encoder
        self.encoder = nn.Sequential(
            *layers
        )
        self.vf_encoder = nn.Sequential(
            *layers
        )

        # core rnn
        self.hidden_state_size = self.custom_config["model_arch_args"]["hidden_state_size"]

        if self.custom_config["model_arch_args"]["core_arch"] == "gru":
            self.rnn = nn.GRU(input_dim, self.hidden_state_size, batch_first=True)
        elif self.custom_config["model_arch_args"]["core_arch"] == "lstm":
            self.rnn = nn.LSTM(input_dim, self.hidden_state_size, batch_first=True)
        else:
            raise ValueError()

        # action branch and value branch
        self.action_branch = nn.Linear(self.hidden_state_size, num_outputs)
        self.value_branch = nn.Linear(self.input_dim, 1)

        # Holds the current "base" output (before logits layer).
        self._features = None

        # record the custom config
        self.n_agents = self.custom_config["num_agents"]
        self.q_flag = False

        self.actors = [self.encoder, self.rnn, self.action_branch]
        self.actor_initialized_parameters = self.actor_parameters()

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
        B = self._features.shape[0]
        L = self._features.shape[1]
        # Compute the unmasked logits.
        if "conv_layer" in self.custom_config["model_arch_args"]:
            x = self.inputs.reshape(-1, self.inputs.shape[2], self.inputs.shape[3], self.inputs.shape[4]).permute(0, 3,
                                                                                                                  1, 2)
            x = self.vf_encoder(x)
            x = torch.mean(x, (2, 3))
            x = x.reshape(self.inputs.shape[0], self.inputs.shape[1], -1)
        else:
            x = self.vf_encoder(self.inputs)

        if self.q_flag:
            return torch.reshape(self.value_branch(x), [B * L, -1])
        else:
            return torch.reshape(self.value_branch(x), [-1])

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
        self.inputs = inputs

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
            raise ValueError("rnn core_arch wrong: {}".format(self.custom_config["model_arch_args"]["core_arch"]))

    def actor_parameters(self):
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.actors))

    def critic_parameters(self):
        return list(self.value_branch.parameters())

    def sample(self, obs, training_batch, sample_num):
        indices = torch.multinomial(torch.arange(len(obs)), sample_num, replacement=True)
        training_batch = training_batch.copy()
        training_batch['obs']['obs'] = training_batch['obs']['obs'][indices]
        if 'action_mask' in training_batch['obs']:
            training_batch['obs']['action_mask'] = training_batch['obs']['action_mask'][indices]

        return self(training_batch)
