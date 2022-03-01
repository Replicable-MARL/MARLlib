import gym
import numpy as np
import ray.rllib.evaluation.postprocessing as rllib_post
import torch
import torch.nn as nn
from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer, validate_config as ppo_validate_config
from ray.rllib.agents.ppo.ppo_tf_policy import KLCoeffMixin
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, KLCoeffMixin as TorchKLCoeffMixin, \
    ppo_surrogate_loss as original_ppo_torch_loss
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule, LearningRateSchedule, LearningRateSchedule as TorchLR, \
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable
from ray.rllib.utils.torch_ops import explained_variance, convert_to_torch_tensor
from ray.rllib.utils.typing import Dict, List, ModelConfigDict, TensorType
from ray.rllib.agents.ppo.ppo import PPOTrainer, PPOTFPolicy, validate_config as PPO_valid, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_tf_policy import KLCoeffMixin, \
    ppo_surrogate_loss as tf_loss
from ray.rllib.agents.ppo.ppo_torch_policy import KLCoeffMixin as TorchKLCoeffMixin, ppo_surrogate_loss as torch_loss
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.evaluation.postprocessing import adjust_nstep


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        if self.config["framework"] != "torch":
            raise NotImplementedError("Error")
            self.compute_central_vf = make_tf_callable(self.get_session())(self.model.central_value_function)
        else:
            self.compute_central_vf = self.model.central_value_function


def concat_cc_process(policy, sample_batch, other_agent_batches, odim, adim, other_info_dim):
    """Concat the neighbors' observations"""
    for index in range(sample_batch.count):
        neighbours = sample_batch['infos'][index]["neighbours"]
        for n_count, n_name in enumerate(neighbours):
            if n_count >= policy.config["num_neighbours"]:
                break
            n_count = n_count
            if n_name in other_agent_batches and \
                    (index < len(other_agent_batches[n_name][1][SampleBatch.CUR_OBS])):
                if policy.config["counterfactual"]:
                    start = odim + n_count * other_info_dim
                    sample_batch["centralized_critic_obs"][index, start: start + odim] = \
                        other_agent_batches[n_name][1][SampleBatch.CUR_OBS][index]
                    sample_batch["centralized_critic_obs"][index, start + odim: start + odim + adim] = \
                        other_agent_batches[n_name][1][SampleBatch.ACTIONS][index]
                else:
                    sample_batch["centralized_critic_obs"][index, n_count * odim: (n_count + 1) * odim] = \
                        other_agent_batches[n_name][1][SampleBatch.CUR_OBS][index]
    return sample_batch


def mean_field_cc_process(policy, sample_batch, other_agent_batches, odim, adim, other_info_dim):
    """Average the neighbors' observations"""
    for index in range(sample_batch.count):
        neighbours = sample_batch['infos'][index]["neighbours"]
        neighbours_distance = sample_batch['infos'][index]["neighbours_distance"]
        obs_list = []
        act_list = []
        for n_count, (n_name, n_dist) in enumerate(zip(neighbours, neighbours_distance)):
            if n_dist > policy.config["mf_nei_distance"]:
                continue
            if n_name in other_agent_batches and \
                    (index < len(other_agent_batches[n_name][1][SampleBatch.CUR_OBS])):
                obs_list.append(other_agent_batches[n_name][1][SampleBatch.CUR_OBS][index])
                act_list.append(other_agent_batches[n_name][1][SampleBatch.ACTIONS][index])
        if len(obs_list) > 0:
            sample_batch["centralized_critic_obs"][index, odim:odim + odim] = np.mean(obs_list, axis=0)
            if policy.config["counterfactual"]:
                sample_batch["centralized_critic_obs"][index, odim + odim:odim + odim + adim] = np.mean(act_list,
                                                                                                        axis=0)
    return sample_batch


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):
    pytorch = policy.config["framework"] == "torch"
    assert pytorch
    _ = sample_batch[SampleBatch.INFOS]  # touch

    # ===== Grab other's observation and actions to compute the per-agent's centralized values =====
    if (pytorch and hasattr(policy, "compute_central_vf")) or (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None

        o = sample_batch[SampleBatch.CUR_OBS]
        odim = sample_batch[SampleBatch.CUR_OBS].shape[1]
        other_info_dim = odim
        adim = sample_batch[SampleBatch.ACTIONS].shape[1]
        if policy.config["counterfactual"]:
            other_info_dim += adim

        sample_batch["centralized_critic_obs"] = np.zeros(
            (o.shape[0], policy.config["centralized_critic_obs_dim"]), dtype=sample_batch[SampleBatch.CUR_OBS].dtype
        )
        sample_batch["centralized_critic_obs"][:, :odim] = sample_batch[SampleBatch.CUR_OBS]

        if policy.config["fuse_mode"] == "concat":
            sample_batch = concat_cc_process(policy, sample_batch, other_agent_batches, odim, adim, other_info_dim)
        elif policy.config["fuse_mode"] == "mf":
            sample_batch = mean_field_cc_process(
                policy, sample_batch, other_agent_batches, odim, adim, other_info_dim
            )
        else:
            raise ValueError("Unknown fuse mode: {}".format(policy.config["fuse_mode"]))

        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
            convert_to_torch_tensor(sample_batch["centralized_critic_obs"], policy.device)
        ).cpu().detach().numpy()
    else:
        # Policy hasn't been initialized yet, use zeros.
        _ = sample_batch[SampleBatch.INFOS]  # touch
        o = sample_batch[SampleBatch.CUR_OBS]
        sample_batch["centralized_critic_obs"] = np.zeros(
            (o.shape[0], policy.config["centralized_critic_obs_dim"]), dtype=o.dtype
        )
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    if "DDPG" in str(policy.__class__): # MADDPG
        ## copied from postprocess_nstep_and_prio in DDPGTorchPolicy
        if policy.config["n_step"] > 1:
            adjust_nstep(policy.config["n_step"], policy.config["gamma"], sample_batch)

        # Create dummy prio-weights (1.0) in case we don't have any in
        # the batch.
        PRIO_WEIGHTS = "weights"
        if PRIO_WEIGHTS not in sample_batch:
            sample_batch[PRIO_WEIGHTS] = np.ones_like(sample_batch[SampleBatch.REWARDS])

        # Prioritize on the worker side.
        if sample_batch.count > 0 and policy.config["worker_side_prioritization"]:
            td_errors = policy.compute_td_error(
                sample_batch[SampleBatch.OBS], sample_batch[SampleBatch.ACTIONS],
                sample_batch[SampleBatch.REWARDS], sample_batch[SampleBatch.NEXT_OBS],
                sample_batch[SampleBatch.DONES], sample_batch[PRIO_WEIGHTS])
            new_priorities = (np.abs(convert_to_numpy(td_errors)) +
                              policy.config["prioritized_replay_eps"])
            sample_batch[PRIO_WEIGHTS] = new_priorities

    else:  # MAAC MAPPO
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

    # # ===== Compute the centralized values' advantage =====
    # completed = sample_batch["dones"][-1]
    # if completed:
    #     last_r = 0.0
    # else:
    #     last_r = sample_batch[SampleBatch.VF_PREDS][-1]
    #
    # train_batch = compute_advantages(
    #     sample_batch, last_r, policy.config["gamma"], policy.config["lambda"], use_gae=policy.config["use_gae"]
    # )
    # return train_batch


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic_ppo(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = tf_loss if not policy.config["framework"] == "torch" else torch_loss

    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(train_batch["centralized_critic_obs"])
    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)
    model.value_function = vf_saved
    return loss


def setup_tf_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTFPolicy (w/o ValueNetworkMixin).
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"], config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def setup_torch_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTorchPolicy  (w/o ValueNetworkMixin).
    TorchKLCoeffMixin.__init__(policy, config)
    TorchEntropyCoeffSchedule.__init__(policy, config["entropy_coeff"], config["entropy_coeff_schedule"])
    TorchLR.__init__(policy, config["lr"], config["lr_schedule"])


def central_vf_stats_ppo(policy, train_batch):
    # Report the explained variance of the central value function.
    return {
        "value_targets": torch.mean(train_batch[Postprocessing.VALUE_TARGETS]),
        "advantage_mean": torch.mean(train_batch[Postprocessing.ADVANTAGES]),
        "advantages_min": torch.min(train_batch[Postprocessing.ADVANTAGES]),
        "advantages_max": torch.max(train_batch[Postprocessing.ADVANTAGES]),
        "central_value_mean": torch.mean(policy._central_value_out),
        "central_value_min": torch.min(policy._central_value_out),
        "central_value_max": torch.max(policy._central_value_out),
        "cur_kl_coeff": policy.kl_coeff,
        "cur_lr": policy.cur_lr,
        "total_loss": torch.mean(
            torch.stack(policy.get_tower_stats("total_loss"))),
        "policy_loss": torch.mean(
            torch.stack(policy.get_tower_stats("mean_policy_loss"))),
        "vf_loss": torch.mean(
            torch.stack(policy.get_tower_stats("mean_vf_loss"))),
        "vf_explained_var": torch.mean(
            torch.stack(policy.get_tower_stats("vf_explained_var"))),
        "kl": torch.mean(torch.stack(policy.get_tower_stats("mean_kl_loss"))),
        "entropy": torch.mean(
            torch.stack(policy.get_tower_stats("mean_entropy"))),
        "entropy_coeff": policy.entropy_coeff,
    }


def get_centralized_critic_obs_dim(
        observation_space_shape, action_space_shape, counterfactual, num_neighbours, fuse_mode
):
    """Get the centralized critic"""
    if fuse_mode == "concat":
        pass
    elif fuse_mode == "mf":
        num_neighbours = 1
    else:
        raise ValueError("Unknown fuse mode: ", fuse_mode)
    num_neighbours += 1
    centralized_critic_obs_dim = num_neighbours * observation_space_shape.shape[0]
    if counterfactual:
        # Do not include ego action!
        centralized_critic_obs_dim += (num_neighbours - 1) * action_space_shape.shape[0]
    return centralized_critic_obs_dim


def make_model(policy, obs_space, action_space, config):
    """Overwrite the model config here!"""
    policy.config["exclude_act_dim"] = np.prod(action_space.shape)
    config["model"]["centralized_critic_obs_dim"] = config["centralized_critic_obs_dim"]
    dist_class, logit_dim = ModelCatalog.get_action_dist(action_space, config["model"])
    return ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=logit_dim,
        model_config=config["model"],
        framework=config["framework"]
    )


def vf_preds_fetches(policy, input_dict, state_batches, model, action_dist):
    return dict()
