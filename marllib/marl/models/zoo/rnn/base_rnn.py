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

from ray.rllib.utils.torch_ops import FLOAT_MIN
import numpy as np
from typing import Dict, List
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch, \
    TensorType
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from functools import reduce
from marllib.marl.models.zoo.encoder.base_encoder import BaseEncoder

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class BaseRNN(TorchRNN, nn.Module):

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
        self.n_agents = self.custom_config["num_agents"]
        self.activation = model_config.get("fcnet_activation")

        # encoder
        self.p_encoder = BaseEncoder(model_config, self.full_obs_space)
        self.vf_encoder = BaseEncoder(model_config, self.full_obs_space)

        # core rnn
        self.hidden_state_size = self.custom_config["model_arch_args"]["hidden_state_size"]

        if self.custom_config["model_arch_args"]["core_arch"] == "gru":
            self.rnn = nn.GRU(self.p_encoder.output_dim, self.hidden_state_size, batch_first=True)
        elif self.custom_config["model_arch_args"]["core_arch"] == "lstm":
            self.rnn = nn.LSTM(self.p_encoder.output_dim, self.hidden_state_size, batch_first=True)
        else:
            raise ValueError(
                "should be either gru or lstm, got {}".format(self.custom_config["model_arch_args"]["core_arch"]))

        # action branch and value branch
        self.p_branch = SlimFC(
            in_size=self.hidden_state_size,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None)
        self.vf_branch = SlimFC(
            in_size=self.vf_encoder.output_dim,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        # Holds the current "base" output (before logits layer).
        self._features = None

        # record the custom config
        self.n_agents = self.custom_config["num_agents"]
        self.q_flag = False

        self.actors = [self.p_encoder, self.rnn, self.p_branch]
        self.actor_initialized_parameters = self.actor_parameters()

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        if self.custom_config["model_arch_args"]["core_arch"] == "gru":
            h = [
                self.vf_branch._model._modules["0"].weight.new(1, self.hidden_state_size).zero_().squeeze(0),
            ]
        else:  # lstm
            h = [
                self.vf_branch._model._modules["0"].weight.new(1, self.hidden_state_size).zero_().squeeze(0),
                self.vf_branch._model._modules["0"].weight.new(1, self.hidden_state_size).zero_().squeeze(0)
            ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        B = self._features.shape[0]
        L = self._features.shape[1]
        # Compute the unmasked logits.
        x = self.vf_encoder(self.inputs)

        if self.q_flag:
            return torch.reshape(self.vf_branch(x), [B * L, -1])
        else:
            return torch.reshape(self.vf_branch(x), [-1])

    @override(ModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                hidden_state: List[TensorType],
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
        output, hidden_state = self.forward_rnn(inputs, hidden_state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])

        if self.custom_config["mask_flag"]:
            output = output + inf_mask

        return output, hidden_state

    @override(TorchRNN)
    def forward_rnn(self, inputs, hidden_state, seq_lens):
        self.inputs = inputs

        x = self.p_encoder(self.inputs)

        if self.custom_config["model_arch_args"]["core_arch"] == "gru":
            self._features, h = self.rnn(x, torch.unsqueeze(hidden_state[0], 0))
            logits = self.p_branch(self._features)
            return logits, [torch.squeeze(h, 0)]

        elif self.custom_config["model_arch_args"]["core_arch"] == "lstm":
            self._features, [h, c] = self.rnn(
                x, [torch.unsqueeze(hidden_state[0], 0),
                    torch.unsqueeze(hidden_state[1], 0)])
            logits = self.p_branch(self._features)
            return logits, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

        else:
            raise ValueError("rnn core_arch wrong: {}".format(self.custom_config["model_arch_args"]["core_arch"]))

    def actor_parameters(self):
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.actors))

    def critic_parameters(self):
        return list(self.vf_branch.parameters())

    # def sample(self, obs, training_batch, sample_num):
    #     indices = torch.multinomial(torch.arange(len(obs)), sample_num, replacement=True)
    #     training_batch = training_batch.copy()
    #     training_batch['obs']['obs'] = training_batch['obs']['obs'][indices]
    #     if 'action_mask' in training_batch['obs']:
    #         training_batch['obs']['action_mask'] = training_batch['obs']['action_mask'][indices]
    #
    #     return self(training_batch)
