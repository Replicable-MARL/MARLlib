from ray.rllib.agents.ppo.ppo_torch_policy import ValueNetworkMixin
from ray.rllib.utils.torch_ops import apply_grad_clipping, sequence_mask
import logging
import gym
from typing import Dict, Tuple

import ray
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    concat_multi_gpu_td_errors, huber_loss, l2_loss
from ray.rllib.utils.typing import TrainerConfigDict, TensorType, \
    LocalOptimizer, GradInfoDict
from SMAC.util.mappo_tools import centralized_critic_postprocessing, CentralizedValueMixin
from ray.rllib.agents.a3c.a3c_tf_policy import actor_critic_loss as tf_loss
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from gym.spaces.box import Box
from ray.rllib.evaluation.postprocessing import adjust_nstep
from ray.rllib.utils.numpy import convert_to_numpy
import numpy as np
from ray.rllib.utils.framework import try_import_tf, try_import_torch

torch, nn = try_import_torch()


def loss_with_central_critic_coma(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = coma_loss

    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(
        train_batch["self_obs"], train_batch["state"], train_batch["opponent_action"])

    # recording data
    policy._central_value_out = model.value_function()

    loss = func(policy, model, dist_class, train_batch)
    model.value_function = vf_saved

    return loss


def central_vf_stats_coma(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy._central_value_out)
    }


def coma_loss(policy: Policy, model: ModelV2,
              dist_class: ActionDistribution,
              train_batch: SampleBatch) -> TensorType:
    logits, _ = model.from_batch(train_batch)
    values = model.value_function()

    if policy.is_recurrent():
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask_orig = sequence_mask(train_batch[SampleBatch.SEQ_LENS],
                                  max_seq_len)
        valid_mask = torch.reshape(mask_orig, [-1])
    else:
        valid_mask = torch.ones_like(values, dtype=torch.bool)

    dist = dist_class(logits, model)
    log_probs = dist.logp(train_batch[SampleBatch.ACTIONS]).reshape(-1)

    # here the coma loss & calculate the mean values as baseline:
    select_action_Q_value = values.gather(1, train_batch[SampleBatch.ACTIONS].unsqueeze(1)).squeeze()
    advantages = (select_action_Q_value - torch.mean(values, dim=1)).detach()
    coma_pi_err = -torch.sum(torch.masked_select(log_probs * advantages, valid_mask))

    # Compute coma critic loss.
    if policy.config["use_critic"]:
        value_err = 0.5 * torch.sum(
            torch.pow(
                torch.masked_select(
                    select_action_Q_value.reshape(-1) -
                    train_batch[Postprocessing.VALUE_TARGETS], valid_mask),
                2.0))
    # Ignore the value function.
    else:
        value_err = 0.0

    entropy = torch.sum(torch.masked_select(dist.entropy(), valid_mask))

    total_loss = (coma_pi_err + value_err * policy.config["vf_loss_coeff"] -
                  entropy * policy.config["entropy_coeff"])

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["entropy"] = entropy
    model.tower_stats["pi_err"] = coma_pi_err
    model.tower_stats["value_err"] = value_err

    return total_loss


def loss_and_entropy_stats(policy: Policy,
                           train_batch: SampleBatch) -> Dict[str, TensorType]:
    return {
        "policy_entropy": torch.mean(
            torch.stack(policy.get_tower_stats("entropy"))),
        "policy_loss": torch.mean(
            torch.stack(policy.get_tower_stats("pi_err"))),
        "vf_loss": torch.mean(
            torch.stack(policy.get_tower_stats("value_err"))),
    }


def coma_model_value_predictions(
        policy: Policy, input_dict: Dict[str, TensorType], state_batches,
        model: ModelV2,
        action_dist: ActionDistribution) -> Dict[str, TensorType]:
    return {SampleBatch.VF_PREDS: model.value_function()}


def torch_optimizer(policy: Policy,
                    config: TrainerConfigDict) -> LocalOptimizer:
    return torch.optim.Adam(policy.model.parameters(), lr=config["lr"])


def setup_mixins(policy: Policy, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 config: TrainerConfigDict) -> None:
    """Call all mixin classes' constructors before PPOPolicy initialization.

    Args:
        policy (Policy): The Policy object.
        obs_space (gym.spaces.Space): The Policy's observation space.
        action_space (gym.spaces.Space): The Policy's action space.
        config (TrainerConfigDict): The Policy's config.
    """
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)


def centralized_critic_postprocessing_coma(policy,
                                           sample_batch,
                                           other_agent_batches=None,
                                           episode=None):
    pytorch = policy.config["framework"] == "torch"
    self_obs_dim = policy.config["self_obs_dim"]
    state_dim = policy.config["state_dim"]

    # action_dim = policy.action_space.n
    if (pytorch and hasattr(policy, "compute_central_vf")) or \
            (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None
        sample_batch["self_obs"] = sample_batch['obs'][:, :self_obs_dim]
        sample_batch["state"] = sample_batch['obs'][:, self_obs_dim:self_obs_dim + state_dim]

        # overwrite default VF prediction with the central VF
        if pytorch:
            sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
                convert_to_torch_tensor(
                    sample_batch["self_obs"], policy.device),
                convert_to_torch_tensor(
                    sample_batch["state"], policy.device), ) \
                .cpu().detach().numpy().mean(1)
        else:  # not used
            sample_batch["vf_preds"] = policy.compute_central_vf(
                sample_batch["obs"], sample_batch["opponent_obs"],
                sample_batch["opponent_action"])
    else:
        # Policy hasn't been initialized yet, use zeros.
        o = sample_batch[SampleBatch.CUR_OBS]
        sample_batch["self_obs"] = np.zeros((o.shape[0], self_obs_dim),
                                            dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
        sample_batch["state"] = np.zeros((o.shape[0], state_dim),
                                         dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch["vf_preds"][-1]

    sample_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])

    return sample_batch


# based on a3c torch policy
COMATorchPolicy = build_policy_class(
    name="COMATorchPolicy",
    framework="torch",
    get_default_config=lambda: ray.rllib.agents.a3c.a3c.DEFAULT_CONFIG,
    loss_fn=coma_loss,
    stats_fn=loss_and_entropy_stats,
    postprocess_fn=centralized_critic_postprocessing_coma,
    extra_action_out_fn=coma_model_value_predictions,  # may have bug as we have action dim vf function output in coma
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=torch_optimizer,
    before_loss_init=setup_mixins,
    mixins=[ValueNetworkMixin],
)
