"""
Implement TRPO and HATRPO in Ray Rllib
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
from ray.rllib.utils.torch_ops import apply_grad_clipping
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from marl.algos.utils.setup_utils import setup_torch_mixins
from marl.algos.utils.centralized_critic_hetero import (
    trpo_post_process,
    value_normalizer,
)

from marl.algos.utils.trust_regions import TrustRegionUpdator

from ray.rllib.examples.centralized_critic import CentralizedValueMixin
from icecream import ic

tf1, tf, tfv = try_import_tf()
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

    # policy_loss = reduce_mean_valid(logp_ratio * advantages)

    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)
    action_kl = prev_action_dist.kl(curr_action_dist)

    if policy.config["use_critic"]:
        prev_value_fn_out = train_batch[SampleBatch.VF_PREDS] #
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

    # if loss.isnan() or mean_vf_loss.isnan():
    #     print('loss error, find nan')
    #     ic(loss)
    #     ic(mean_vf_loss)

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


# def extra_gradients(policy, gradients) -> None:
#     policy.trpo_updator.update()

def apply_gradients(policy, gradients) -> None:
    # print('\nstep into apply updater!')
    policy.trpo_updator.update()


TRPOTorchPolicy = PPOTorchPolicy.with_updates(
        name="TRPO-TorchPolicy",
        get_default_config=lambda: PPO_CONFIG,
        postprocess_fn=trpo_post_process,
        loss_fn=trpo_loss_fn,
        before_init=setup_torch_mixins,
        # optimizer_fn=make_happo_optimizers,
        # extra_grad_process_fn=apply_grad_clipping,
        apply_gradients_fn=apply_gradients,
        # extra_grad_process_fn=extra_gradients,
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
