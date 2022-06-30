from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.preprocessors import get_preprocessor

torch, nn = try_import_torch()


class JointQ_RNN(TorchModelV2, nn.Module):
    """The default GRU model for Joint Q."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        custom_config = model_config["custom_model_config"]

        # currently only support gru cell
        if custom_config["model_arch_args"]["core_arch"] != "gru":
            raise ValueError()

        self.obs_size = _get_size(obs_space)

        # encoder
        layers = []
        if "fc_layer" in custom_config["model_arch_args"]:
            input_dim = self.obs_size
            for i in range(custom_config["model_arch_args"]["fc_layer"]):
                out_dim = custom_config["model_arch_args"]["out_dim_fc_{}".format(i)]
                fc_layer = nn.Linear(input_dim, out_dim)
                layers.append(fc_layer)
                input_dim = out_dim
        elif "conv_layer" in custom_config["model_arch_args"]:
            input_dim = obs_space.shape[2]
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

        self.hidden_state_size = custom_config["model_arch_args"]["hidden_state_size"]
        self.rnn = nn.GRUCell(input_dim, self.hidden_state_size)
        self.q_value = nn.Linear(self.hidden_state_size, num_outputs)

        self.n_agents = custom_config["num_agents"]
        # record the custom config
        self.custom_config = custom_config
        if custom_config["global_state_flag"]:
            state_dim = custom_config["space_obs"]["state"].shape
        else:
            state_dim = custom_config["space_obs"]["obs"].shape
        self.raw_state_dim = state_dim


    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
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
        h_in = hidden_state[0].reshape(-1, self.hidden_state_size)
        h = self.rnn(x, h_in)
        q = self.q_value(h)
        return q, [h]


def _get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size
