"""TRPO policy implemented in PyTorch."""
import numpy as np
import torch

from nnrl.optim import build_optimizer


from ray.rllib import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.utils import override
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from raylab.options import configure, option
from raylab.policy import TorchPolicy, learner_stats
from raylab.policy.action_dist import WrapStochasticPolicy
from raylab.utils.dictionaries import get_keys
from raylab.utils.explained_variance import explained_variance