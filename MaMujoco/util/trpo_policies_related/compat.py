"""Wrappers and adapters for compatibility with RLlib's API."""
from typing import Dict, List

from gym import Space
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import TensorType
from ray.rllib.utils.typing import ModelConfigDict
from torch import nn


class WrapRawModule(TorchModelV2, nn.Module):
    """Wrapper around a basic PyTorch module.

    Mostly for compatibility with ray>=1.1.0.
    """

    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        module: nn.Module,
        num_outputs: int,
        model_config: ModelConfigDict,
    ):
        # pylint:disable=too-many-arguments
        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name="PyTorchModule",
        )
        nn.Module.__init__(self)

        self.module = module

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        pass

    def value_function(self) -> TensorType:
        pass

    def import_from_h5(self, h5_file: str) -> None:
        pass
