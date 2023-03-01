from ray.rllib.utils.torch_ops import FLOAT_MIN
from functools import reduce
import copy
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, SlimConv2d, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List
from marllib.marl.models.zoo.encoder.base_encoder import Base_Encoder

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
        self.activation = model_config.get("fcnet_activation")

        # encoder
        self.p_encoder = Base_Encoder(model_config, self.full_obs_space)
        self.vf_encoder = Base_Encoder(model_config, self.full_obs_space)

        self.p_branch = SlimFC(
            in_size=self.p_encoder.output_dim,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        # self.vf_encoder = nn.Sequential(*copy.deepcopy(layers))
        self.vf_branch = SlimFC(
            in_size=self.vf_encoder.output_dim,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_obs = None

        self.q_flag = False

        self.actors = [self.p_encoder, self.p_branch]
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

        self.inputs = flat_inputs
        self._features = self.p_encoder(self.inputs)

        output = self.p_branch(self._features)

        if self.custom_config["mask_flag"]:
            output = output + inf_mask

        return output, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        B = self._features.shape[0]
        x = self.vf_encoder(self.inputs)

        if self.q_flag:
            return torch.reshape(self.vf_branch(x), [B, -1])
        else:
            return torch.reshape(self.vf_branch(x), [-1])

    def actor_parameters(self):
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.actors))

    def critic_parameters(self):
        return list(self.vf_branch.parameters())
