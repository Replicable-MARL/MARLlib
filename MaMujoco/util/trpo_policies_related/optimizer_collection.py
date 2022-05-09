"""Standard PyTorch optimizer interface for TorchPolicies."""
import contextlib
from collections import OrderedDict
from collections.abc import MutableMapping
from typing import Iterator

from torch.optim import Optimizer


class OptimizerCollection(MutableMapping):
    """A collection of PyTorch `Optimizer`s with names."""

    def __init__(self):
        self._optimizers = OrderedDict()

    def __setitem__(self, key: str, value: Optimizer):
        """Adds an optimizer to the collection.

        Args:
            key: the name of the optimizer
            value: the optimizer instance

        Raises:
            ValueError: if the value to insert is not an optimizer
        """
        assert key not in self._optimizers, f"'{key}' optimizer already in collection"
        if not isinstance(value, Optimizer):
            raise ValueError("'{type(value).__name__}' is not an Optimizer instance")

        self._optimizers[key] = value

    def __getitem__(self, key: str) -> Optimizer:
        """Gets an optimizer from the collection.

        Args:
            key: the name of the optimizer

        Returns:
            The optimizer instance
        """
        if key not in self._optimizers:
            raise KeyError(f"'{key}' optimizer not in collection")
        return self._optimizers[key]

    def __delitem__(self, key: str):
        """Removes an optimizer from the collection.

        Args:
            key: the name of the optimizer
        """
        if key not in self._optimizers:
            raise KeyError(f"'{key}' optimizer not in collection")
        del self._optimizers[key]

    def __iter__(self) -> Iterator[str]:
        """Returns an iterator over the optimizer names in the collection."""
        return iter(self._optimizers.keys())

    def __len__(self) -> int:
        """Returns the number of optimizers in the collection."""
        return len(self._optimizers)

    @contextlib.contextmanager
    def optimize(self, name: str):
        """Nullify grads before context and step optimizer afterwards."""
        optimizer = self[name]
        optimizer.zero_grad()
        yield
        optimizer.step()

    def state_dict(self) -> dict:
        """Returns the state of each optimizer in the collection."""
        return {k: v.state_dict() for k, v in self.items()}

    def load_state_dict(self, state_dict: dict):
        """Loads the state of each optimizer.

        Args:
            state_dict: optimizer collection state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        for name, optim in self._optimizers.items():
            optim.load_state_dict(state_dict[name])
