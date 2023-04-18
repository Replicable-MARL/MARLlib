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
from ray.rllib.utils.torch_ops import explained_variance
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.typing import TensorType
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, KLCoeffMixin
from ray.rllib.utils.torch_ops import apply_grad_clipping
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from marllib.marl.algos.utils.setup_utils import setup_torch_mixins
from marllib.marl.algos.utils.centralized_critic_hetero import (
    contain_global_obs,
    hatrpo_post_process,
)

from marllib.marl.algos.utils.trust_regions import TrustRegionUpdator
from marllib.marl.algos.utils.heterogeneous_updateing import update_m_advantage, get_each_agent_train, get_mask_and_reduce_mean

from ray.rllib.examples.centralized_critic import CentralizedValueMixin
from marllib.marl.algos.utils.setup_utils import get_device

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


logger = logging.getLogger(__name__)


def get_trpo_loss(reduce_mean, mask, logp_ratio, advantages):
    if reduce_mean == torch.mean:
        loss = torch.sum(logp_ratio * advantages, dim=-1, keepdim=True).mean()
    else:
        loss = (torch.sum(logp_ratio * advantages, dim=-1, keepdim=True) *
                mask).sum() / mask.sum()

    return loss


def hatrpo_loss_fn(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for Heterogeneous Agent TRPO
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

    # advantages = train_batch[Postprocessing.ADVANTAGES]
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    vf_saved = model.value_function

    if contain_global_obs(train_batch):
        opp_action_in_cc = policy.config["model"]["custom_model_config"]["opp_action_in_cc"]
        model.value_function = lambda: policy.model.central_value_function(train_batch["state"],
                                                                           train_batch[
                                                                               "opponent_actions"] if opp_action_in_cc else None)
    # if not contain_opponent_info:
    #     updater = TrustRegionUpdator
    # else:

    _, reduce_mean_valid, curr_action_dist = get_mask_and_reduce_mean(model, train_batch, dist_class)

    curr_entropy = curr_action_dist.entropy()

    # Compute a value function loss.
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

    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)
    action_kl = prev_action_dist.kl(curr_action_dist)

    m_advantage = train_batch[Postprocessing.ADVANTAGES]

    loss = 0

    agent_num = 1

    model.train()
    for i, iter_agent_info in enumerate(get_each_agent_train(model, policy, dist_class, train_batch)):
        iter_model, iter_dist_class, iter_train_batch, iter_mask, \
            iter_reduce_mean, iter_actions, iter_policy, iter_prev_action_logp = iter_agent_info

        iter_logits, iter_state = iter_model(iter_train_batch)
        iter_current_action_dist = iter_dist_class(iter_logits, iter_model)
        iter_prev_action_logp = iter_train_batch[SampleBatch.ACTION_LOGP]
        iter_logp_ratio = torch.exp(iter_current_action_dist.logp(iter_actions) - iter_prev_action_logp)

        iter_loss = get_trpo_loss(
            reduce_mean=iter_reduce_mean,
            mask=iter_mask,
            logp_ratio=iter_logp_ratio,
            advantages=m_advantage
        )

        loss += iter_loss

        trust_region_updator = TrustRegionUpdator(
            model=iter_model,
            dist_class=iter_dist_class,
            train_batch=iter_train_batch,
            adv_targ=m_advantage.to(device=get_device()),
            initialize_policy_loss=iter_loss.to(device=get_device()),
        )

        trust_region_updator.update(update_critic=False)

        m_advantage = update_m_advantage(
            iter_model=iter_model,
            iter_dist_class=iter_dist_class,
            iter_train_batch=iter_train_batch,
            iter_actions=iter_actions,
            m_advantage=m_advantage,
            iter_prev_action_logp=iter_prev_action_logp
        )

        agent_num += 1

    model.value_function = vf_saved
    # recovery the value function.

    total_loss = reduce_mean_valid(policy.kl_coeff * action_kl +
                                                  policy.config["vf_loss_coeff"] * vf_loss -
                                                  policy.entropy_coeff * curr_entropy
                                                  )
    total_loss = total_loss.to(device=get_device())

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    mean_kl_loss = reduce_mean_valid(action_kl)
    mean_policy_loss = -1 * loss / agent_num
    mean_entropy = reduce_mean_valid(curr_entropy)

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], model.value_function())
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss


HAPTRPOTorchPolicy = PPOTorchPolicy.with_updates(
        name="HAPPOTorchPolicy",
        get_default_config=lambda: PPO_CONFIG,
        postprocess_fn=hatrpo_post_process,
        loss_fn=hatrpo_loss_fn,
        before_init=setup_torch_mixins,
        extra_grad_process_fn=apply_grad_clipping,
        mixins=[
            EntropyCoeffSchedule, KLCoeffMixin,
            CentralizedValueMixin, LearningRateSchedule,
        ])


def get_policy_class_hatrpo(config_):
    if config_["framework"] == "torch":
        return HAPTRPOTorchPolicy


HATRPOTrainer = PPOTrainer.with_updates(
    name="HATRPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_hatrpo,
)