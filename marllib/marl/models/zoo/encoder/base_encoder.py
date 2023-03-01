from ray.rllib.utils.torch_ops import FLOAT_MIN
from functools import reduce
import copy
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, SlimConv2d, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List

torch, nn = try_import_torch()


class Base_Encoder(nn.Module):
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
            input_dim = obs_space['obs'].shape[0]
            for out_dim in self.encoder_layer_dim:
                layers.append(
                    SlimFC(in_size=input_dim,
                           out_size=out_dim,
                           initializer=normc_initializer(1.0),
                           activation_fn=self.activation))
                input_dim = out_dim
        elif "conv_layer" in self.custom_config["model_arch_args"]:
            input_dim = obs_space['obs'].shape[2]
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

        self.output_dim = input_dim  # record
        self.encoder = nn.Sequential(*layers)

    def forward(self, inputs) -> (TensorType, List[TensorType]):
        B = inputs.shape[0]
        # Compute the unmasked logits.
        if "conv_layer" in self.custom_config["model_arch_args"]:
            if self.custom_config["model_arch_args"]["core_arch"] in ["gru", "lstm"] and len(inputs.shape) > 4:
                L = inputs.shape[1]
                inputs = inputs.reshape((-1,) + inputs.shape[2:])
                # inputs.reshape(B*L, )
                x = self.encoder(inputs.permute(0, 3, 1, 2))
                x = torch.mean(x, (2, 3))
                output = x.reshape((B, L, -1))

            else:
                x = self.encoder(inputs.permute(0, 3, 1, 2))
                output = torch.mean(x, (2, 3))
        else:
            self.inputs = inputs.reshape(inputs.shape[0], -1)
            output = self.encoder(inputs)

        return output
