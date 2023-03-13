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

from ray.rllib.models.torch.misc import SlimFC, SlimConv2d, normc_initializer
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List

torch, nn = try_import_torch()


class CentralizedEncoder(nn.Module):
    """Generic fully connected network."""

    def __init__(
            self,
            model_config,
            obs_space
    ):
        nn.Module.__init__(self)

        # decide the model arch
        self.custom_config = model_config["custom_model_config"]
        self.activation = model_config.get("fcnet_activation")
        self.num_agents = self.custom_config["num_agents"]

        # encoder
        layers = []
        if "fc_layer" in self.custom_config["model_arch_args"]:
            if "encode_layer" in self.custom_config["model_arch_args"]:
                encode_layer = self.custom_config["model_arch_args"]["encode_layer"]
                encoder_layer_dim = encode_layer.split("-")
                encoder_layer_dim = [int(i) for i in encoder_layer_dim]
            else:  # default config
                encoder_layer_dim = []
                for i in range(self.custom_config["model_arch_args"]["fc_layer"]):
                    out_dim = self.custom_config["model_arch_args"]["out_dim_fc_{}".format(i)]
                    encoder_layer_dim.append(out_dim)

            self.encoder_layer_dim = encoder_layer_dim
            if "state" not in obs_space.spaces:
                input_dim = self.num_agents * obs_space['obs'].shape[0]
            else:
                input_dim = obs_space['state'].shape[0] + obs_space['obs'].shape[0]
            for out_dim in self.encoder_layer_dim:
                layers.append(
                    SlimFC(in_size=input_dim,
                           out_size=out_dim,
                           initializer=normc_initializer(1.0),
                           activation_fn=self.activation))
                input_dim = out_dim
        elif "conv_layer" in self.custom_config["model_arch_args"]:
            if "state" not in obs_space.spaces:
                self.state_dim = obs_space["obs"].shape
                self.state_dim_last = self.state_dim[-1]
                input_dim = obs_space['obs'].shape[2]
            else:
                self.state_dim = obs_space["state"].shape
                self.state_dim_last = obs_space["state"].shape[2] + obs_space["obs"].shape[2]
                input_dim = obs_space['state'].shape[2] + obs_space['obs'].shape[2]

            for i in range(self.custom_config["model_arch_args"]["conv_layer"]):
                layers.append(
                    SlimConv2d(
                        in_channels=input_dim,
                        out_channels=self.custom_config["model_arch_args"]["out_channel_layer_{}".format(i)],
                        kernel=self.custom_config["model_arch_args"]["kernel_size_layer_{}".format(i)],
                        stride=self.custom_config["model_arch_args"]["stride_layer_{}".format(i)],
                        padding=self.custom_config["model_arch_args"]["padding_layer_{}".format(i)],
                        activation_fn=self.activation
                    )
                )
                pool_f = nn.MaxPool2d(kernel_size=self.custom_config["model_arch_args"]["pool_size_layer_{}".format(i)])
                layers.append(pool_f)

                input_dim = self.custom_config["model_arch_args"]["out_channel_layer_{}".format(i)]

        else:
            raise ValueError("fc_layer/conv layer not in model arch args")

        if "state" not in obs_space.spaces and "conv_layer" in self.custom_config["model_arch_args"]:
            self.output_dim = input_dim * self.num_agents  # record
        else:
            self.output_dim = input_dim  # record
        self.encoder = nn.Sequential(*layers)

    def forward(self, inputs) -> (TensorType, List[TensorType]):

        # Compute the unmasked logits.
        if "conv_layer" in self.custom_config["model_arch_args"]:
            x = inputs.reshape(-1, self.state_dim[0], self.state_dim[1], self.state_dim_last)
            x = self.encoder(x.permute(0, 3, 1, 2))
            output = torch.mean(x, (2, 3))
        else:
            inputs = inputs.reshape(inputs.shape[0], -1)
            output = self.encoder(inputs)

        return output
