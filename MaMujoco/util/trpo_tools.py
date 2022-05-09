import torch
from torch import nn
import numpy as np

from MaMujoco.util.trpo_utilities import (
    conjugate_gradient,
    hessian_vector_product,
    line_search,
    flat_grad,
    learner_stats,
    get_keys,
    explained_variance,
)

from ray.rllib import SampleBatch
from ray.rllib.policy import TorchPolicy
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union, \
    TYPE_CHECKING
from ray.rllib.utils import override
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing, compute_gae_for_sample_batch
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def postprocess():
    pass


def loss_fun():
    pass


def extra_gradient():
    pass