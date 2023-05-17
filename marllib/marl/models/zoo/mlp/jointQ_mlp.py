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

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.misc import SlimFC, SlimConv2d, normc_initializer
from marllib.marl.models.zoo.encoder.base_encoder import BaseEncoder

torch, nn = try_import_torch()


class JointQMLP(TorchModelV2, nn.Module):
    """sneaky gru-like mlp"""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        self.custom_config = model_config["custom_model_config"]
        # decide the model arch
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.n_agents = self.custom_config["num_agents"]

        # currently only support gru cell
        if self.custom_config["model_arch_args"]["core_arch"] != "mlp":
            raise ValueError("core_arch can only be MLP")

        self.activation = model_config.get("fcnet_activation")

        # encoder
        self.encoder = BaseEncoder(model_config, {'obs': self.full_obs_space})
        self.hidden_state_size = self.custom_config["model_arch_args"]["hidden_state_size"]
        self.mlp = nn.Linear(self.encoder.output_dim, self.hidden_state_size)
        self.q_value = SlimFC(
            in_size=self.hidden_state_size,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        # record the custom config
        if self.custom_config["global_state_flag"]:
            state_dim = self.custom_config["space_obs"]["state"].shape
        else:
            state_dim = self.custom_config["space_obs"]["obs"].shape
        self.raw_state_dim = state_dim

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        return [
            self.q_value._model._modules["0"].weight.new(self.n_agents,
                                                         self.hidden_state_size).zero_().squeeze(0)
        ]

    @override(ModelV2)
    def forward(self, input_dict, hidden_state, seq_lens):
        inputs = input_dict["obs_flat"].float()
        if len(self.full_obs_space.shape) == 3:  # 3D
            inputs = inputs.reshape((-1,) + self.full_obs_space.shape)
        x = self.encoder(inputs)
        h = hidden_state[0].reshape(-1, self.hidden_state_size)  # fake a hidden state no use
        x = self.mlp(x)
        q = self.q_value(x)
        return q, [h]


def _get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size
