"""Action distribution for compatibility with RLlib's interface."""
from abc import ABCMeta, abstractmethod

from nnrl.nn.actor import DeterministicPolicy, StochasticPolicy
from ray.rllib.models.action_dist import ActionDistribution
from torch import nn


class IncompatibleDistClsError(Exception):
    """Exception raised for incompatible action distribution and NN module.

    Args:
        dist_cls: Action distribution class
        module: NN module
        err: AssertionError explaining the reason why distribution and
            module are incompatible

    Attributes:
        message: Human-readable text explaining what caused the incompatibility
    """

    def __init__(self, dist_cls: type, module: nn.Module, err: Exception):
        # pylint:disable=unused-argument
        msg = (
            f"Action distribution type {dist_cls} is incompatible"
            " with NN module of type {type(module)}. Reason:\n"
            "    {err}"
        )
        super().__init__(msg)
        self.message = msg


class BaseActionDist(ActionDistribution, metaclass=ABCMeta):
    """Base class for TorchPolicy action distributions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = self.model.module

    @classmethod
    def check_model_compat(cls, model: nn.Module):
        """Assert the given NN module is compatible with the distribution.

        Raises:
            IncompatibleDistClsError: If `model` is incompatible with the
                distribution class
        """
        try:
            cls._check_model_compat(model)
        except AssertionError as err:
            raise IncompatibleDistClsError(cls, model, err) from err

    @classmethod
    @abstractmethod
    def _check_model_compat(cls, model: nn.Module):
        pass


class WrapStochasticPolicy(BaseActionDist):
    """Wraps an nn.Module with a stochastic actor and its inputs.

    Expects actor to be an instance of StochasticPolicy.
    """

    # pylint:disable=abstract-method
    valid_actor_cls = StochasticPolicy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sampled_logp = None

    def sample(self):
        action, logp = self.module.actor.sample(**self.inputs)
        self._sampled_logp = logp
        return action, logp

    def deterministic_sample(self):
        return self.module.actor.deterministic(**self.inputs)

    def sampled_action_logp(self):
        return self._sampled_logp

    def logp(self, x):
        return self.module.actor.log_prob(value=x, **self.inputs)

    def entropy(self):
        return self.module.actor.entropy(**self.inputs)

    @classmethod
    def _check_model_compat(cls, model):
        assert hasattr(model, "actor"), f"NN model {type(model)} has no actor attribute"
        assert isinstance(model.actor, cls.valid_actor_cls), (
            f"Expected actor to be an instance of {cls.valid_actor_cls};"
            " found {type(model.actor)} instead."
        )


class WrapDeterministicPolicy(BaseActionDist):
    """Wraps an nn.Module with a deterministic actor and its inputs.

    Expects actor to be an instance of DeterministicPolicy.
    """

    # pylint:disable=abstract-method
    valid_actor_cls = valid_behavior_cls = DeterministicPolicy

    def sample(self):
        action = self.module.behavior(**self.inputs)
        return action, None

    def deterministic_sample(self):
        return self.module.actor(**self.inputs), None

    def sampled_action_logp(self):
        return None

    def logp(self, x):
        return None

    @classmethod
    def _check_model_compat(cls, model: nn.Module):
        assert hasattr(model, "actor"), f"NN model {type(model)} has no actor attribute"
        assert isinstance(model.actor, cls.valid_actor_cls), (
            f"Expected actor to be an instance of {cls.valid_actor_cls};"
            " found {type(model.actor)} instead."
        )

        assert hasattr(model, "behavior"), "NN has no behavior attribute"
        assert isinstance(model.actor, cls.valid_behavior_cls), (
            f"Expected behavior to be an instance of {cls.valid_behavior_cls};"
            " found {type(model.behavior)} instead."
        )
