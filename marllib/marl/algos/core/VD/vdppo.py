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

from ray.rllib.models.action_dist import ActionDistribution
from typing import List, Type, Union
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import explained_variance, sequence_mask
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, KLCoeffMixin, ValueNetworkMixin
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from marllib.marl.algos.utils.mixing_critic import MixingValueMixin, value_mixing_postprocessing

torch, nn = try_import_torch()


# value decomposition based ppo loss
def value_mix_ppo_surrogate_loss(
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
    MixingValueMixin.__init__(policy)

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

    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS],
                                  model)

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

        # add mixing_function
        opponent_vf_preds = convert_to_torch_tensor(train_batch["opponent_vf_preds"])
        vf_pred = value_fn_out.unsqueeze(1)
        all_vf_pred = torch.cat((vf_pred, opponent_vf_preds), 1)
        state = convert_to_torch_tensor(train_batch["state"])
        value_tot = model.mixing_value(all_vf_pred, state)

        vf_loss1 = torch.pow(
            value_tot - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_clipped = prev_value_fn_out + torch.clamp(
            value_tot - prev_value_fn_out, -policy.config["vf_clip_param"],
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


VDPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="VDPPOTorchPolicy",
    get_default_config=lambda: PPO_CONFIG,
    postprocess_fn=value_mixing_postprocessing,
    loss_fn=value_mix_ppo_surrogate_loss,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, MixingValueMixin
    ])


def get_policy_class_vdppo(config_):
    if config_["framework"] == "torch":
        return VDPPOTorchPolicy


VDPPOTrainer = PPOTrainer.with_updates(
    name="VDPPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_vdppo,
)
