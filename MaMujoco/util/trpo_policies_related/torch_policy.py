"""Base for all PyTorch policies."""
import textwrap
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import torch
from gym.spaces import Space
from ray.rllib.utils.torch_ops import convert_to_torch_tensor as convert_to_tensor
# from nnrl.utils import convert_to_tensor
from ray.rllib import Policy, SampleBatch
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.models.modelv2 import flatten, restore_original_dimensions
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils import override
from ray.rllib.utils.torch_ops import convert_to_non_torch_type, convert_to_torch_tensor
from ray.rllib.utils.typing import ModelGradients, TensorType
from ray.tune.logger import pretty_print
from torch import Tensor, nn
from MaMujoco.util.options import RaylabOptions, configure, option

from .action_dist import BaseActionDist
from .compat import WrapRawModule
from .modules import get_module
from .optimizer_collection import OptimizerCollection


@configure
@option("env", default=None)
@option("env_config/", allow_unknown_subkeys=True)
@option("explore", default=True)
@option(
    "exploration_config/", allow_unknown_subkeys=True, override_all_if_type_changes=True
)
@option("framework", default="torch")
@option("gamma", default=0.99)
@option("num_workers", default=0)
@option("seed", default=None)
@option("worker_index", default=0)
@option("normalize_actions", default=True)
@option("clip_actions", default=False)
@option(
    "module/",
    help="Type and config of the PyTorch NN module.",
    allow_unknown_subkeys=True,
    override_all_if_type_changes=True,
)
@option(
    "optimizer/",
    help="Config dict for PyTorch optimizers.",
    allow_unknown_subkeys=True,
)
@option("compile", False, help="Whether to optimize the policy's backend")
class TorchPolicy(Policy):
    """A Policy that uses PyTorch as a backend.

    Attributes:
        observation_space: Space of possible observation inputs
        action_space: Space of possible action outputs
        config: Policy configuration
        dist_class: Action distribution class for computing actions. Must be set
            by subclasses before calling `__init__`.
        device: Device in which the parameter tensors reside. All input samples
            will be converted to tensors and moved to this device
        module: The policy's neural network module. Should be compilable to
            TorchScript
        optimizers: The optimizers bound to the neural network (or submodules)
        options: Configuration object for this class
    """

    observation_space: Space
    action_space: Space
    dist_class: Type[BaseActionDist]
    config: dict
    global_config: dict
    device: torch.device
    model: WrapRawModule
    module: nn.Module
    optimizers: OptimizerCollection
    options: RaylabOptions = RaylabOptions()

    def __init__(self, observation_space: Space, action_space: Space, config: dict):
        # Allow subclasses to set `dist_class` before calling init
        action_dist: Optional[Type[BaseActionDist]] = getattr(self, "dist_class", None)
        super().__init__(observation_space, action_space, self._build_config(config))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module = self._make_module(observation_space, action_space, self.config)
        self.module.to(self.device)
        self.model = WrapRawModule(
            self.observation_space,
            self.action_space,
            self.module,
            num_outputs=1,
            model_config=self.config["module"],
        )

        self.optimizers = self._make_optimizers()

        # === Policy attributes ===
        self.dist_class: Type[BaseActionDist] = action_dist
        self.dist_class.check_model_compat(self.module)
        self.framework = "torch"  # Needed to create exploration
        self.exploration = self._create_exploration()

    # ==========================================================================
    # PublicAPI
    # ==========================================================================

    @property
    def pull_from_global(self) -> Set[str]:
        """Keys to pull from global configuration.

        Configurations passed down from caller (usually by the trainer) that are
        not under the `policy` config.
        """
        return {
            "env",
            "env_config",
            "explore",
            # "exploration_config",
            "gamma",
            "num_workers",
            "seed",
            "worker_index",
            "normalize_actions",
            "clip_actions",
        }

    def compile(self):
        """Optimize modules with TorchScript.

        Warnings:
            This action cannot be undone.
        """
        self.module = torch.jit.script(self.module)

    @torch.no_grad()
    @override(Policy)
    def compute_actions(
        self,
        obs_batch: Union[List[TensorType], TensorType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorType], TensorType] = None,
        prev_reward_batch: Union[List[TensorType], TensorType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List[MultiAgentEpisode]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        # pylint:disable=too-many-arguments,too-many-locals
        explore = explore if explore is not None else self.config["explore"]
        timestep = timestep if timestep is not None else self.global_timestep

        input_dict = self.lazy_tensor_dict(
            SampleBatch({SampleBatch.CUR_OBS: obs_batch, "is_training": False})
        )
        if prev_action_batch is not None:
            input_dict[SampleBatch.PREV_ACTIONS] = prev_action_batch
        if prev_reward_batch is not None:
            input_dict[SampleBatch.PREV_REWARDS] = prev_reward_batch
        state_batches = convert_to_torch_tensor(state_batches or [], device=self.device)

        # Call the exploration before_compute_actions hook.
        self.exploration.before_compute_actions(timestep=timestep)

        unpacked = unpack_observations(
            input_dict, self.observation_space, self.framework
        )
        state_out = state_batches

        # pylint:disable=not-callable
        action_dist = self.dist_class({"obs": unpacked["obs"]}, self.model)
        # pylint:enable=not-callable
        actions, logp = self.exploration.get_exploration_action(
            action_distribution=action_dist, timestep=timestep, explore=explore
        )
        input_dict[SampleBatch.ACTIONS] = actions

        # Add default and custom fetches.
        extra_fetches = {}
        if logp is not None:
            extra_fetches[SampleBatch.ACTION_PROB] = logp.exp()
            extra_fetches[SampleBatch.ACTION_LOGP] = logp

        return convert_to_non_torch_type((actions, state_out, extra_fetches))

    def _get_default_view_requirements(self):
        # Add extra fetch keys to view requirements so that they're available
        # for training
        return {
            SampleBatch.ACTION_PROB: ViewRequirement(used_for_training=True),
            SampleBatch.ACTION_LOGP: ViewRequirement(used_for_training=True),
            **super()._get_default_view_requirements(),
        }

    @torch.no_grad()
    @override(Policy)
    def compute_log_likelihoods(
        self,
        actions: Union[List[TensorType], TensorType],
        obs_batch: Union[List[TensorType], TensorType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Optional[Union[List[TensorType], TensorType]] = None,
        prev_reward_batch: Optional[Union[List[TensorType], TensorType]] = None,
        actions_normalized: bool = True,
    ) -> TensorType:
        # pylint:disable=too-many-arguments
        input_dict = self.lazy_tensor_dict(
            SampleBatch({SampleBatch.CUR_OBS: obs_batch, SampleBatch.ACTIONS: actions})
        )
        if prev_action_batch is not None:
            input_dict[SampleBatch.PREV_ACTIONS] = prev_action_batch
        if prev_reward_batch is not None:
            input_dict[SampleBatch.PREV_REWARDS] = prev_reward_batch

        dist_inputs, _ = self.module(
            unpack_observations(input_dict, self.observation_space, self.framework),
            state_batches,
            self.convert_to_tensor([1]),
        )
        # pylint:disable=not-callable
        action_dist = self.dist_class(dist_inputs, self.module)
        # pylint:enable=not-callable
        log_likelihoods = action_dist.logp(input_dict[SampleBatch.ACTIONS])
        return log_likelihoods

    @override(Policy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        if not self.config["env_config"].get("time_aware", False):
            hit_limit = sample_batch[SampleBatch.INFOS][-1].get("TimeLimit.truncated")
            env_done = sample_batch[SampleBatch.DONES][-1]
            sample_batch[SampleBatch.DONES][-1] = False if hit_limit else env_done
        return sample_batch

    @override(Policy)
    def get_weights(self) -> dict:
        return {
            "module": convert_to_non_torch_type(self.module.state_dict()),
            # Optimizer state dicts don't store tensors, only ids
            "optimizers": self.optimizers.state_dict(),
        }

    @override(Policy)
    def set_weights(self, weights: dict):
        self.module.load_state_dict(
            convert_to_torch_tensor(weights["module"], device=self.device)
        )
        # Optimizer state dicts don't store tensors, only ids
        self.optimizers.load_state_dict(weights["optimizers"])

    def convert_to_tensor(self, arr) -> Tensor:
        """Convert an array to a PyTorch tensor in this policy's device.

        Args:
            arr (array_like): object which can be converted using `np.asarray`
        """
        return convert_to_tensor(arr, self.device)

    def lazy_tensor_dict(self, sample_batch: SampleBatch) -> SampleBatch:
        """Convert a sample batch into a dictionary of lazy tensors.

        The sample batch is wrapped with a UsageTrackingDict to convert array-
        likes into tensors upon querying.

        Args:
            sample_batch: the sample batch to convert

        Returns:
            A dictionary which intercepts key queries to lazily convert arrays
            to tensors.
        """
        tensor_batch = sample_batch.copy(shallow=True)
        tensor_batch.set_get_interceptor(self.convert_to_tensor)
        return tensor_batch

    def __repr__(self):
        name = self.__class__.__name__
        args = [f"{self.observation_space},", f"{self.action_space},"]

        config = pretty_print(self.config).rstrip("\n")
        if "\n" in config:
            config = textwrap.indent(config, " " * 2)
            config = "{\n" + config + "\n}"

            args += [config]
            args_repr = "\n".join(args)
            args_repr = textwrap.indent(args_repr, " " * 2)
            constructor = f"{name}(\n{args_repr}\n)"
        else:
            args += [config]
            args_repr = " ".join(args[1:-1])
            constructor = f"{name}({args_repr})"
        return constructor

    def apply_gradients(self, gradients: ModelGradients) -> None:
        pass

    def compute_gradients(
        self, postprocessed_batch: SampleBatch
    ) -> Tuple[ModelGradients, Dict[str, TensorType]]:
        pass

    # ==========================================================================
    # InternalAPI
    # ==========================================================================

    def _build_config(self, config: dict) -> dict:
        if not self.options.all_options_set:
            raise RuntimeError(
                f"{type(self).__name__} still has configs to be set."
                " Did you call `configure` as the last decorator?"
            )

        passed = config.get("policy", {}).copy()
        passed.update({k: config[k] for k in self.pull_from_global if k in config})
        new = self.options.merge_defaults_with(passed)
        return new

    @staticmethod
    def _make_module(obs_space: Space, action_space: Space, config: dict) -> nn.Module:
        """Build the PyTorch nn.Module to be used by this policy.

        Args:
            obs_space: the observation space for this policy
            action_space: the action_space for this policy
            config: the user config containing the 'module' key

        Returns:
            A neural network module.
        """
        return get_module(obs_space, action_space, config["module"])

    def _make_optimizers(self) -> OptimizerCollection:
        """Build PyTorch optimizers to use.

        The result will be set as the policy's optimizer collection.

        The user should update the optimizer collection (mutable mapping)
        returned by the base implementation.

        Returns:
            A mapping from names to optimizer instances
        """
        # pylint:disable=no-self-use
        return OptimizerCollection()

    # ==========================================================================
    # Unimplemented Policy methods
    # ==========================================================================

    def get_num_samples_loaded_into_buffer(self, buffer_index: int = 0) -> int:
        pass

    def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0):
        pass

    def load_batch_into_buffer(self, batch: SampleBatch, buffer_index: int = 0) -> int:
        pass

    def export_model(self, export_dir: str, onnx: Optional[int] = None) -> None:
        pass

    def export_checkpoint(self, export_dir):
        pass

    def import_model_from_h5(self, import_file):
        pass


def unpack_observations(input_dict, observation_space: Space, framework: str):
    """Cast observations to original space and add a separate flattened view."""
    restored = input_dict.copy()
    restored["obs"] = restore_original_dimensions(
        input_dict["obs"], observation_space, framework
    )
    if len(input_dict["obs"].shape) > 2:
        restored["obs_flat"] = flatten(input_dict["obs"], framework)
    else:
        restored["obs_flat"] = input_dict["obs"]
    return restored
