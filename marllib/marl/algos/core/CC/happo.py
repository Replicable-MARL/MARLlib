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
Implement HAPPO algorithm based on Rlib original PPO.
__author__: minquan
__data__: March-29-2022
"""

from typing import List, Type, Union
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.torch_ops import explained_variance
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.utils.torch_ops import apply_grad_clipping
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from marllib.marl.algos.utils.setup_utils import setup_torch_mixins
from marllib.marl.algos.utils.centralized_critic_hetero import (
    add_all_agents_gae,
)
from ray.rllib.examples.centralized_critic import CentralizedValueMixin
from marllib.marl.algos.utils.setup_utils import get_device
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, KLCoeffMixin
import torch
from marllib.marl.algos.utils.heterogeneous_updateing import update_m_advantage, get_each_agent_train


def happo_surrogate_loss(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for Heterogeneous Agent PPO Objective.
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

    reduce_mean_valid, mean_entropy, mean_kl_loss = None, None, None

    agent_num = 1
    policies_loss = 0

    m_advantage = train_batch[Postprocessing.ADVANTAGES]

    for i, iter_train_info in enumerate(get_each_agent_train(model, policy, dist_class, train_batch)):
        iter_model, iter_dist_class, iter_train_batch, iter_mask, \
            iter_reduce_mean, iter_actions, iter_policy, iter_prev_action_logp = iter_train_info

        iter_model.train()

        iter_prev_action_dist = iter_dist_class(iter_train_batch[SampleBatch.ACTION_DIST_INPUTS], iter_model)

        iter_logits, iter_state = iter_model(iter_train_batch)
        iter_current_action_dist = iter_dist_class(iter_logits, iter_model)
        logp_ratio = torch.exp(iter_current_action_dist.logp(iter_actions) - iter_prev_action_logp)

        iter_action_kl = iter_prev_action_dist.kl(iter_current_action_dist)
        iter_mean_kl_loss = iter_reduce_mean(iter_action_kl)

        iter_curr_entropy = iter_current_action_dist.entropy()
        iter_mean_entropy = iter_reduce_mean(iter_curr_entropy)

        if iter_model == model:
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

        policies_loss += iter_surrogate_loss

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

        m_advantage = update_m_advantage(
            iter_model=iter_model,
            iter_train_batch=iter_train_batch,
            iter_actions=iter_actions,
            m_advantage=m_advantage,
            iter_dist_class=iter_dist_class,
            iter_prev_action_logp=iter_prev_action_logp
        )

        agent_num += 1

    mean_policy_loss = policies_loss / agent_num

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
