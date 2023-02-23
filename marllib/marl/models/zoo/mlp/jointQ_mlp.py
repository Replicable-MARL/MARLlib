from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer
torch, nn = try_import_torch()


class JointQ_MLP(TorchModelV2, nn.Module):
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
            raise ValueError()

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
        self.activation = model_config.get("fcnet_activation")
        self.obs_size = _get_size(obs_space)

        # encoder
        layers = []
        if "fc_layer" in self.custom_config["model_arch_args"]:
            input_dim = self.obs_size
            for out_dim in self.encoder_layer_dim:
                layers.append(
                    SlimFC(in_size=input_dim,
                           out_size=out_dim,
                           initializer=normc_initializer(1.0),
                           activation_fn=self.activation))
                input_dim = out_dim
        elif "conv_layer" in self.custom_config["model_arch_args"]:
            input_dim = obs_space.shape[2]
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

        self.hidden_state_size = self.custom_config["model_arch_args"]["hidden_state_size"]
        self.mlp = nn.Linear(input_dim, self.hidden_state_size)
        self.q_value = nn.Linear(self.hidden_state_size, num_outputs)

        self.n_agents = self.custom_config["num_agents"]
        # record the custom config
        self.custom_config = self.custom_config
        if self.custom_config["global_state_flag"]:
            state_dim = self.custom_config["space_obs"]["state"].shape
        else:
            state_dim = self.custom_config["space_obs"]["obs"].shape
        self.raw_state_dim = state_dim


    @override(ModelV2)
    def get_initial_state(self):
        # fake a hidden ste
        return [
            self.q_value.weight.new(self.n_agents,
                                    self.hidden_state_size).zero_().squeeze(0)
        ]

    @override(ModelV2)
    def forward(self, input_dict, hidden_state, seq_lens):
        inputs = input_dict["obs_flat"].float()
        if "conv_layer" in self.custom_config["model_arch_args"]:
            x = inputs.reshape(-1, self.raw_state_dim[0], self.raw_state_dim[1], self.raw_state_dim[2]).permute(0, 3, 1, 2)
            x = self.encoder(x)
            x = torch.mean(x, (2, 3))
            x = x.reshape(inputs.shape[0], -1)
        else:
            x = self.encoder(inputs)
        h = hidden_state[0].reshape(-1, self.hidden_state_size) # fake a hidden state no use
        x = self.mlp(x)
        q = self.q_value(x)
        return q, [h]


def _get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size
