"""
Implement HAPPO algorithm based on Rlib original PPO.
__author__: minquan
__data__: March-29-2022
"""


import gym
import logging
from typing import Dict, List, Type, Union

import ray
from ray.rllib.agents.ppo.ppo_tf_policy import setup_config
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, \
    Postprocessing
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule, \
    LearningRateSchedule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    explained_variance, sequence_mask
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from ray.rllib.utils.torch_ops import convert_to_torch_tensor

import argparse
import numpy as np
from gym.spaces import Discrete
import os

import ray
from ray import tune
from ray.rllib.agents.maml.maml_torch_policy import KLCoeffMixin as TorchKLCoeffMixin
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import (
    PPOTFPolicy,
    KLCoeffMixin,
    ppo_surrogate_loss as tf_loss,
)
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.examples.models.centralized_critic_models import (
    CentralizedCriticModel,
    TorchCentralizedCriticModel,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import (
    LearningRateSchedule as TorchLR,
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.numpy import convert_to_numpy
from MaMujoco.util.vda2c_tools import MixingValueMixin
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

"""
def postprocess_fn(policy, sample_batch, other_agent_batches, episode):
    agents = ["agent_1", "agent_2", "agent_3"]  # simple example of 3 agents
    global_obs_batch = np.stack(
        [other_agent_batches[agent_id][1]["obs"] for agent_id in agents],
        axis=1)
    # add the global obs and global critic value
    sample_batch["global_obs"] = global_obs_batch
    sample_batch["central_vf"] = self.sess.run(
        self.critic_network, feed_dict={"obs": global_obs_batch})
    return sample_batch

"""
from ray.rllib.examples.centralized_critic import centralized_critic_postprocessing


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.

GLOBAL_NEED_COLLECT = [SampleBatch.OBS, SampleBatch.ACTIONS, SampleBatch.ACTION_LOGP]
GLOBAL_PREFIX = 'GLOBAL_'


def add_other_agent_info(agents_batch: dict, key: str):
    # get other-agents information by specific key

    _POLICY_INDEX, _BATCH_INDEX = 0, 1

    return np.stack([
        agents_batch[agent_id][_BATCH_INDEX][key] for agent_id in agents_batch],
        axis=1
    )


def get_global_name(key):
    # converts a key to global format

    return f'{GLOBAL_PREFIX}{key}'


def add_another_agent_and_gae(policy, sample_batch, other_agent_batches=None, episode=None):
    train_batch = compute_gae_for_sample_batch(policy, sample_batch, other_agent_batches, episode)

    if other_agent_batches:

        for key in GLOBAL_NEED_COLLECT:
            train_batch[get_global_name(key)] = add_other_agent_info(agents_batch=other_agent_batches, key=key)

    return train_batch


def contain_global_obs(train_batch):
    return any(key.startswith(GLOBAL_PREFIX) for key in train_batch)


def ppo_surrogate_loss(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for Proximal Policy Objective.
    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.
    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major())
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean

    if contain_global_obs(train_batch):
        print('contain global observation!')


    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) -
        train_batch[SampleBatch.ACTION_LOGP])
    action_kl = prev_action_dist.kl(curr_action_dist)
    mean_kl_loss = reduce_mean_valid(action_kl)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES] * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"],
            1 + policy.config["clip_param"]))
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)

    # Compute a value function loss.
    if policy.config["use_critic"]:
        prev_value_fn_out = train_batch[SampleBatch.VF_PREDS]
        value_fn_out = model.value_function()
        vf_loss1 = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_clipped = prev_value_fn_out + torch.clamp(
            value_fn_out - prev_value_fn_out, -policy.config["vf_clip_param"],
            policy.config["vf_clip_param"])
        vf_loss2 = torch.pow(
            vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_loss = torch.max(vf_loss1, vf_loss2)
        mean_vf_loss = reduce_mean_valid(vf_loss)
    # Ignore the value function.
    else:
        vf_loss = mean_vf_loss = 0.0

    total_loss = reduce_mean_valid(-surrogate_loss +
                                   policy.kl_coeff * action_kl +
                                   policy.config["vf_loss_coeff"] * vf_loss -
                                   policy.entropy_coeff * curr_entropy)

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], model.value_function())
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss
