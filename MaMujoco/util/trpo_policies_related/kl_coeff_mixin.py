# pylint:disable=missing-docstring
# pylint: enable=missing-docstring
from abc import abstractmethod
from dataclasses import dataclass, field


@dataclass
class AdaptiveKLCoeffSpec:
    """Adaptive schedule for KL Divergence regularization."""

    initial_coeff: float = 0.2
    desired_kl: float = 0.01
    adaptation_coeff: float = 2.0
    threshold: float = 1.5
    curr_coeff: float = field(init=False)

    def __post_init__(self):
        self.curr_coeff = self.initial_coeff

    def adapt(self, kl_div):
        """Apply PPO rule to update current KL coeff based on latest KL divergence."""
        if kl_div < self.desired_kl / self.threshold:
            self.curr_coeff /= self.adaptation_coeff
        elif kl_div > self.desired_kl * self.threshold:
            self.curr_coeff *= self.adaptation_coeff


class AdaptiveKLCoeffMixin:
    """Adds adaptive KL penalty as in PPO."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kl_coeff_spec = AdaptiveKLCoeffSpec(**self.config["kl_schedule"])

    def update_kl_coeff(self, sample_batch):
        """Update KL penalty based on observed divergence between successive policies.

        Arguments:
            sample_batch (SampleBatch): batch of data gathered by the old policy

        Returns:
            A dictionary with a single entry corresponding to the average KL divergence.
        """
        kl_div = self._kl_divergence(sample_batch)
        self._kl_coeff_spec.adapt(kl_div)
        return {"policy_kl_div": kl_div}

    @property
    def curr_kl_coeff(self):
        """Return current KL coefficient."""
        return self._kl_coeff_spec.curr_coeff

    @abstractmethod
    def _kl_divergence(self, sample_batch):
        """Compute the empirical average KL divergence between new and old policies.

        Arguments:
            sample_batch (SampleBatch): batch of data gathered by the old policy

        Returns:
            A scalar equal to the average KL divergence between new and old policies.
        """
