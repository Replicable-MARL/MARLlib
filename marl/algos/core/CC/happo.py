"""
Implement HAPPO algorithm based on Rlib original PPO.
__author__: minquan
__data__: March-29-2022
"""

import random
from typing import Dict, List, Type, Union, Tuple

from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.torch_ops import explained_variance, sequence_mask
import numpy as np
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_gae_for_sample_batch
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import TrainerConfigDict, TensorType, \
    LocalOptimizer
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, ValueNetworkMixin, KLCoeffMixin
from ray.rllib.utils.torch_ops import apply_grad_clipping
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from marl.algos.utils.setup_utils import setup_torch_mixins, get_agent_num
from marl.algos.utils.centralized_critic_hetero import (
    get_global_name,
    add_all_agents_gae,
    global_state_name,
)
from ray.rllib.examples.centralized_critic import CentralizedValueMixin
from marl.algos.utils.setup_utils import get_device
from marl.algos.utils.manipulate_tensor import flat_grad, flat_params
from icecream import ic
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, ValueNetworkMixin, KLCoeffMixin, ppo_surrogate_loss
import torch
import datetime
import re


def get_mask_and_reduce_mean(model, train_batch):
    logits, state = model(train_batch)
    # curr_action_dist = dist_class(logits, model)

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

    return mask, reduce_mean_valid


class IterTrainBatch(SampleBatch):
    def __init__(self, main_train_batch, policy_name):
        self.main_train_batch = main_train_batch
        self.policy_name = policy_name

        self.copy = self.main_train_batch.copy
        self.keys = self.main_train_batch.keys
        self.is_training = self.main_train_batch.is_training

        self.pat = re.compile(r'^state_in_(\d+)')

    def get_state_index(self, string):
        match = self.pat.findall(string)
        if match:
            return match[0]
        else:
            return None

    def __getitem__(self, item):
        """
        Adds an adaptor to get the item.
        Input a key name, it would get the corresponding opponent's key-value
        """
        directly_get = [SampleBatch.SEQ_LENS]

        if item in directly_get:
            return self.main_train_batch[item]
        elif get_global_name(item, self.policy_name) in self.main_train_batch:
            return self.main_train_batch[get_global_name(item, self.policy_name)]
        elif state_index := self.get_state_index(item):
            return self.main_train_batch[global_state_name(state_index, self.policy_name)]

    def __contains__(self, item):
        if item in self.keys() or get_global_name(item, self.policy_name) in self.keys():
            return True
        elif state_index := self.get_state_index(item):
            if global_state_name(state_index, self.policy_name) in self.keys():
                return True

        return False


def happo_surrogate_loss(
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

    CentralizedValueMixin.__init__(policy)

    vf_saved = model.value_function

    # set Global-V-Network
    opp_action_in_cc = policy.config["model"]["custom_model_config"]["opp_action_in_cc"]

    model.value_function = lambda: policy.model.central_value_function(train_batch["state"],
                                                                       train_batch[
                                                                           "opponent_actions"] if opp_action_in_cc else None)

    all_policies_with_names = list(model.other_policies.items()) + [('self', policy)]
    random.shuffle(all_policies_with_names)

    total_policy_loss = 0

    m_advantage = train_batch[Postprocessing.ADVANTAGES]

    reduce_mean_valid, mean_entropy, mean_kl_loss = None, None, None

    for policy_name, iter_policy in all_policies_with_names:
        is_self = (policy_name == 'self')
        iter_model = [iter_policy.model, model][is_self]
        iter_dist_class = [iter_policy.dist_class, dist_class][is_self]
        iter_train_batch = [IterTrainBatch(train_batch, policy_name), train_batch][is_self]
        iter_mask, iter_reduce_mean = get_mask_and_reduce_mean(iter_model, iter_train_batch)

        iter_model.train()

        iter_logits, iter_state = iter_model(iter_train_batch)
        iter_current_action_dist = iter_dist_class(iter_logits, iter_model)

        iter_prev_action_dist = iter_dist_class(iter_train_batch[SampleBatch.ACTION_DIST_INPUTS], iter_model)
        iter_prev_action_logp = iter_train_batch[SampleBatch.ACTION_LOGP]
        iter_actions = iter_train_batch[SampleBatch.ACTIONS]

        logp_ratio = torch.exp(iter_current_action_dist.logp(iter_actions) - iter_prev_action_logp)

        iter_action_kl = iter_prev_action_dist.kl(iter_current_action_dist)
        iter_mean_kl_loss = iter_reduce_mean(iter_action_kl)

        iter_curr_entropy = iter_current_action_dist.entropy()
        iter_mean_entropy = iter_reduce_mean(iter_curr_entropy)

        if policy_name == 'self':
            reduce_mean_valid = iter_reduce_mean
            mean_entropy = iter_mean_entropy
            mean_kl_loss = iter_mean_kl_loss

        iter_surrogate_loss = torch.min(
            m_advantage * logp_ratio,
            m_advantage * torch.clamp(
                logp_ratio, 1 - iter_policy.config["clip_param"], 1 + iter_policy.config["clip_param"]
            )
        )

        iter_surrogate_loss = iter_reduce_mean(iter_surrogate_loss)

        total_policy_loss = total_policy_loss + iter_surrogate_loss

        torch.autograd.set_detect_anomaly(True)

        current_lr = (
            iter_policy.cur_lr / iter_model.custom_config['critic_lr'] * iter_model.custom_config['actor_lr']
        )

        iter_model.update_actor(
            loss=iter_surrogate_loss + iter_policy.entropy_coeff * iter_mean_entropy -
                 iter_policy.kl_coeff * iter_mean_kl_loss,
            lr=current_lr,
            grad_clip=iter_policy.config['grad_clip'],
        )

        with torch.no_grad():
            iter_model.eval()
            iter_new_logits, _ = iter_model(iter_train_batch)
            if torch.any(torch.isnan(iter_new_logits)): ic(iter_new_logits)
            try:
                iter_new_action_dist = iter_dist_class(iter_new_logits, iter_model)
                iter_new_logp_ratio = torch.exp(
                    iter_new_action_dist.logp(iter_actions) -
                    iter_prev_action_logp
                )
            except ValueError as e:
                print(e)
                ic(iter_new_logits)

        m_advantage = iter_new_logp_ratio * m_advantage

    mean_policy_loss = -1 * (total_policy_loss / len(all_policies_with_names))

    if policy.config["use_critic"]:
        prev_value_fn_out = train_batch[SampleBatch.VF_PREDS]  #
        value_fn_out = model.value_function()  # same as values
        vf_loss1 = torch.pow(
            value_fn_out.to(device=get_device()) - train_batch[Postprocessing.VALUE_TARGETS].to(device=get_device()),
            2.0)
        vf_clipped = (prev_value_fn_out + torch.clamp(
            value_fn_out - prev_value_fn_out, -policy.config["vf_clip_param"],
            policy.config["vf_clip_param"])).to(device=get_device())
        vf_loss2 = torch.pow(
            vf_clipped.to(device=get_device()) - train_batch[Postprocessing.VALUE_TARGETS].to(device=get_device()), 2.0)
        vf_loss = torch.max(vf_loss1, vf_loss2).to(device=get_device())
        mean_vf_loss = reduce_mean_valid(vf_loss).to(device=get_device())
    # Ignore the value function.
    else:
        vf_loss = mean_vf_loss = 0.0

    model.value_function = vf_saved
    # recovery the value function.

    value_loss = reduce_mean_valid(policy.config['vf_loss_coeff'] * vf_loss.to(device=get_device()))

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["total_loss"] = value_loss + mean_policy_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS].to(device=get_device()),
        model.value_function().to(device=get_device()))
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return value_loss


HAPPOTorchPolicy = lambda ppo_with_critic: PPOTorchPolicy.with_updates(
    name="HAPPOTorchPolicy",
    get_default_config=lambda: ppo_with_critic,
    postprocess_fn=add_all_agents_gae,
    loss_fn=happo_surrogate_loss,
    before_init=setup_torch_mixins,
    extra_grad_process_fn=apply_grad_clipping,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        CentralizedValueMixin
    ])


def get_policy_class_happo(ppo_with_critic):
    def __inner(config_):
        if config_["framework"] == "torch":
            return HAPPOTorchPolicy(ppo_with_critic)

    return __inner


HAPPOTrainer = lambda ppo_with_critic: PPOTrainer.with_updates(
    name="HAPPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_happo(ppo_with_critic),
)
