from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.misc import SlimFC, SlimConv2d, normc_initializer
from marllib.marl.models.zoo.encoder.base_encoder import Base_Encoder

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

        self.activation = model_config.get("fcnet_activation")

        # encoder
        self.encoder = Base_Encoder(model_config, {'obs': self.full_obs_space})
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
        if len(self.full_obs_space.shape) == 3: # 3D
            inputs = inputs.reshape((-1,) + self.full_obs_space.shape)
        x = self.encoder(inputs)
        h = hidden_state[0].reshape(-1, self.hidden_state_size) # fake a hidden state no use
        x = self.mlp(x)
        q = self.q_value(x)
        return q, [h]


def _get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size
