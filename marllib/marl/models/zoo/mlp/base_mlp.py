from ray.rllib.utils.torch_ops import FLOAT_MIN
from functools import reduce
import copy
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

torch, nn = try_import_torch()


class Base_MLP(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        # decide the model arch
        self.custom_config = model_config["custom_model_config"]
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.n_agents = self.custom_config["num_agents"]

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
        self.obs_size = self.full_obs_space['obs'].shape[0]
        layers = []
        input_dim = self.obs_size

        # Create layers 0 to second-last.
        for out_dim in self.encoder_layer_dim:
            layers.append(
                SlimFC(in_size=input_dim,
                       out_size=out_dim,
                       initializer=normc_initializer(1.0),
                       activation_fn=self.activation))
            input_dim = out_dim

        self.action_encoder = nn.Sequential(*layers)
        self.action_branch = SlimFC(
            in_size=input_dim,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        self.value_encoder = nn.Sequential(*copy.deepcopy(layers))
        self.value_branch = SlimFC(
            in_size=input_dim,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_obs = None

        self.actors = [self.action_encoder, self.action_branch]
        self.actor_initialized_parameters = self.actor_parameters()

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        if self.custom_config["global_state_flag"] or self.custom_config["mask_flag"]:
            flat_inputs = input_dict["obs"]["obs"].float()
            # Convert action_mask into a [0.0 || -inf]-type mask.
            if self.custom_config["mask_flag"]:
                action_mask = input_dict["obs"]["action_mask"]
                inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        else:
            flat_inputs = input_dict["obs"]["obs"].float()

        self._last_obs = flat_inputs.reshape(flat_inputs.shape[0], -1)
        self._features = self.action_encoder(self._last_obs)
        output = self.action_branch(self._features)

        if self.custom_config["mask_flag"]:
            output = output + inf_mask

        return output, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return self.value_branch(
            self.value_encoder(self._last_obs)).squeeze(1)

    def actor_parameters(self):
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.actors))

    def critic_parameters(self):
        return list(self.value_branch.parameters())
