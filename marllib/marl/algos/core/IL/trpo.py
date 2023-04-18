# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Implement TRPO and HATRPO in Ray RLlib
__author__: minquan
__data__: May-15
"""

import logging
from typing import List, Type, Union
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.torch_ops import explained_variance, sequence_mask
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import TensorType
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, KLCoeffMixin
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from marllib.marl.algos.utils.setup_utils import setup_torch_mixins
from marllib.marl.algos.utils.centralized_critic_hetero import trpo_post_process

from marllib.marl.algos.utils.trust_regions import TrustRegionUpdator

from ray.rllib.examples.centralized_critic import CentralizedValueMixin
from icecream import ic

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


def trpo_loss_fn(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for TRPO
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

    advantages = train_batch[Postprocessing.ADVANTAGES]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) -
        train_batch[SampleBatch.ACTION_LOGP]
    )

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

        loss = (torch.sum(logp_ratio * advantages, dim=-1, keepdim=True) *
                mask).sum() / mask.sum()
    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean
        loss = torch.sum(logp_ratio * advantages, dim=-1, keepdim=True).mean()

    curr_entropy = curr_action_dist.entropy()

    # Compute a value function loss.

    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)
    action_kl = prev_action_dist.kl(curr_action_dist)

    if policy.config["use_critic"]:
        prev_value_fn_out = train_batch[SampleBatch.VF_PREDS]  #
        value_fn_out = model.value_function()  # same as values
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

    trust_region_updator = TrustRegionUpdator(
        model=model,
        dist_class=dist_class,
        train_batch=train_batch,
        adv_targ=advantages,
        initialize_policy_loss=loss,
        initialize_critic_loss=mean_vf_loss,
    )

    policy.trpo_updator = trust_region_updator

    total_loss = -loss + reduce_mean_valid(
        policy.kl_coeff * action_kl +
        policy.config["vf_loss_coeff"] * vf_loss -
        policy.entropy_coeff * curr_entropy
    )

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    mean_kl_loss = reduce_mean_valid(action_kl)
    # mean_policy_loss = reduce_mean_valid(-policy_loss)
    mean_entropy = reduce_mean_valid(curr_entropy)

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = -loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], model.value_function())
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss


def apply_gradients(policy, gradients) -> None:
    # print('\nstep into apply updater!')
    policy.trpo_updator.update()


TRPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="TRPO-TorchPolicy",
    get_default_config=lambda: PPO_CONFIG,
    postprocess_fn=trpo_post_process,
    loss_fn=trpo_loss_fn,
    before_init=setup_torch_mixins,
    apply_gradients_fn=apply_gradients,
    mixins=[
        EntropyCoeffSchedule, KLCoeffMixin,
        CentralizedValueMixin, LearningRateSchedule,
    ])


# TRPOTorchPolicy.apply_gradients = apply_gradients


def get_policy_class_trpo(config_):
    if config_["framework"] == "torch":
        return TRPOTorchPolicy


TRPOTrainer = PPOTrainer.with_updates(
    name="TRPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_trpo,
)
