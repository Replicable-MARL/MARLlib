from ray.rllib.agents.ddpg.noop_model import NoopModel, TorchNoopModel
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio, \
    PRIO_WEIGHTS
from ray.rllib.models.catalog import ModelCatalog

from ray.rllib.utils.tf_ops import huber_loss, make_tf_callable
from ray.rllib.utils.typing import TrainerConfigDict, TensorType, \
    LocalOptimizer, ModelGradients
from ray.util.debug import log_once
import logging
import gym
from typing import Dict, Tuple

import ray
from ray.rllib.agents.ddpg.ddpg_tf_policy import build_ddpg_models, \
    get_distribution_inputs_and_class, validate_spaces
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio, \
    PRIO_WEIGHTS
from ray.rllib.agents.sac.sac_torch_policy import TargetNetworkMixin
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDeterministic, \
    TorchDirichlet
from ray.rllib.policy.policy import Policy

from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    concat_multi_gpu_td_errors, huber_loss, l2_loss
from ray.rllib.utils.typing import TrainerConfigDict, TensorType, \
    LocalOptimizer, GradInfoDict
from ray.rllib.agents.ppo.ppo_tf_policy import KLCoeffMixin, \
    ppo_surrogate_loss as tf_loss
from ray.rllib.agents.ppo.ppo_torch_policy import KLCoeffMixin as TorchKLCoeffMixin, ppo_surrogate_loss as torch_loss
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import LearningRateSchedule as TorchLR, \
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from gym.spaces.box import Box
from ray.rllib.evaluation.postprocessing import adjust_nstep
from ray.rllib.utils.numpy import convert_to_numpy
import numpy as np
from ray.rllib.utils.framework import try_import_tf, try_import_torch

from MaMuJoco.model.torch_maddpg import MADDPGTorchModel

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


def maddpg_actor_critic_loss(policy: Policy, model: ModelV2, _,
                             train_batch: SampleBatch) -> TensorType:
    target_model = policy.target_models[model]

    twin_q = policy.config["twin_q"]
    gamma = policy.config["gamma"]
    n_step = policy.config["n_step"]
    use_huber = policy.config["use_huber"]
    huber_threshold = policy.config["huber_threshold"]
    l2_reg = policy.config["l2_reg"]

    input_dict = {
        "obs": train_batch[SampleBatch.CUR_OBS],
        "is_training": True,
    }
    input_dict_next = {
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": True,
    }

    model_out_t, _ = model(input_dict, [], None)

    opponent_num = policy.model.model_config["custom_model_config"]["agent_num"] - 1
    opp_model_out_t_ls = []
    for opponent_index in range(opponent_num):
        opp_input_dict = {
            "obs": train_batch["opponent_obs"][:, opponent_index],
            "is_training": True,
        }
        opp_model_out_t, _ = model(opp_input_dict, [], None)
        opp_model_out_t_ls.append(opp_model_out_t)

    # model_out_tp1, _ = model(input_dict_next, [], None)
    target_model_out_tp1, _ = target_model(input_dict_next, [], None)
    target_opp_model_out_t_ls = []
    for opponent_index in range(opponent_num):
        opp_input_dict = {
            "obs": train_batch["new_opponent_obs"][:, opponent_index],
            "is_training": True,
        }
        target_opp_model_out_t, _ = model(opp_input_dict, [], None)
        target_opp_model_out_t_ls.append(target_opp_model_out_t)

    # Policy network evaluation.
    # prev_update_ops = set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS))
    policy_t = model.get_policy_output(model_out_t)
    # policy_batchnorm_update_ops = list(
    #    set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS)) - prev_update_ops)

    policy_tp1 = target_model.get_policy_output(target_model_out_tp1)

    # Action outputs.
    if policy.config["smooth_target_policy"]:
        target_noise_clip = policy.config["target_noise_clip"]
        clipped_normal_sample = torch.clamp(
            torch.normal(
                mean=torch.zeros(policy_tp1.size()),
                std=policy.config["target_noise"]).to(policy_tp1.device),
            -target_noise_clip, target_noise_clip)

        policy_tp1_smoothed = torch.min(
            torch.max(
                policy_tp1 + clipped_normal_sample,
                torch.tensor(
                    policy.action_space.low,
                    dtype=torch.float32,
                    device=policy_tp1.device)),
            torch.tensor(
                policy.action_space.high,
                dtype=torch.float32,
                device=policy_tp1.device))
    else:
        # No smoothing, just use deterministic actions.
        policy_tp1_smoothed = policy_tp1

    # Q-net(s) evaluation.
    # prev_update_ops = set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS))
    # Q-values for given actions & observations in given current
    # here model_out_t + opp_model_out_t_ls as centralized critic
    q_t = model.get_q_values(model_out_t, opp_model_out_t_ls, train_batch[SampleBatch.ACTIONS])

    # Q-values for current policy (no noise) in given current state
    q_t_det_policy = model.get_q_values(model_out_t, opp_model_out_t_ls, policy_t)

    actor_loss = -torch.mean(q_t_det_policy)

    if twin_q:
        twin_q_t = model.get_twin_q_values(model_out_t,
                                           train_batch[SampleBatch.ACTIONS])
    # q_batchnorm_update_ops = list(
    #     set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS)) - prev_update_ops)

    # Target q-net(s) evaluation.
    # here target_model_out_tp1 + target_opp_model_out_t_ls ascentralized critic
    q_tp1 = target_model.get_q_values(target_model_out_tp1, target_opp_model_out_t_ls,
                                      policy_tp1_smoothed)

    if twin_q:
        twin_q_tp1 = target_model.get_twin_q_values(target_model_out_tp1,
                                                    policy_tp1_smoothed)

    q_t_selected = torch.squeeze(q_t, axis=len(q_t.shape) - 1)
    if twin_q:
        twin_q_t_selected = torch.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
        q_tp1 = torch.min(q_tp1, twin_q_tp1)

    q_tp1_best = torch.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
    q_tp1_best_masked = \
        (1.0 - train_batch[SampleBatch.DONES].float()) * \
        q_tp1_best

    # Compute RHS of bellman equation.
    q_t_selected_target = (train_batch[SampleBatch.REWARDS] +
                           gamma ** n_step * q_tp1_best_masked).detach()

    # Compute the error (potentially clipped).
    if twin_q:
        td_error = q_t_selected - q_t_selected_target
        twin_td_error = twin_q_t_selected - q_t_selected_target
        if use_huber:
            errors = huber_loss(td_error, huber_threshold) \
                     + huber_loss(twin_td_error, huber_threshold)
        else:
            errors = 0.5 * \
                     (torch.pow(td_error, 2.0) + torch.pow(twin_td_error, 2.0))
    else:
        td_error = q_t_selected - q_t_selected_target
        if use_huber:
            errors = huber_loss(td_error, huber_threshold)
        else:
            errors = 0.5 * torch.pow(td_error, 2.0)

    critic_loss = torch.mean(train_batch[PRIO_WEIGHTS] * errors)

    # Add l2-regularization if required.
    if l2_reg is not None:
        for name, var in model.policy_variables(as_dict=True).items():
            if "bias" not in name:
                actor_loss += (l2_reg * l2_loss(var))
        for name, var in model.q_variables(as_dict=True).items():
            if "bias" not in name:
                critic_loss += (l2_reg * l2_loss(var))

    # Model self-supervised losses.
    if policy.config["use_state_preprocessor"]:
        # Expand input_dict in case custom_loss' need them.
        input_dict[SampleBatch.ACTIONS] = train_batch[SampleBatch.ACTIONS]
        input_dict[SampleBatch.REWARDS] = train_batch[SampleBatch.REWARDS]
        input_dict[SampleBatch.DONES] = train_batch[SampleBatch.DONES]
        input_dict[SampleBatch.NEXT_OBS] = train_batch[SampleBatch.NEXT_OBS]
        [actor_loss, critic_loss] = model.custom_loss(
            [actor_loss, critic_loss], input_dict)

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["q_t"] = q_t
    model.tower_stats["actor_loss"] = actor_loss
    model.tower_stats["critic_loss"] = critic_loss
    # TD-error tensor in final stats
    # will be concatenated and retrieved for each individual batch item.
    model.tower_stats["td_error"] = td_error

    # Return two loss terms (corresponding to the two optimizers, we create).
    return actor_loss, critic_loss


def central_vf_stats_ddpg(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy._central_value_out)
    }


def build_maddpg_models(policy: Policy, observation_space: gym.spaces.Space,
                        action_space: gym.spaces.Space,
                        config: TrainerConfigDict) -> ModelV2:

    default_model = TorchNoopModel if config["framework"] == "torch" \
        else NoopModel
    num_outputs = int(np.product(observation_space.shape))

    policy.model = ModelCatalog.get_model_v2(
        obs_space=observation_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface=(MADDPGTorchModel
                         if config["framework"] == "torch" else None),
        default_model=default_model,
        name="maddpg_model",
        actor_hidden_activation=config["actor_hidden_activation"],
        actor_hiddens=config["actor_hiddens"],
        critic_hidden_activation=config["critic_hidden_activation"],
        critic_hiddens=config["critic_hiddens"],
        twin_q=config["twin_q"],
        add_layer_norm=(policy.config["exploration_config"].get("type") ==
                        "ParameterNoise"),
    )

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=observation_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        model_interface=(MADDPGTorchModel
                         if config["framework"] == "torch" else None),
        default_model=default_model,
        name="target_maddpg_model",
        actor_hidden_activation=config["actor_hidden_activation"],
        actor_hiddens=config["actor_hiddens"],
        critic_hidden_activation=config["critic_hidden_activation"],
        critic_hiddens=config["critic_hiddens"],
        twin_q=config["twin_q"],
        add_layer_norm=(policy.config["exploration_config"].get("type") ==
                        "ParameterNoise"),
    )

    return policy.model


def build_maddpg_models_and_action_dist(
        policy: Policy, obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: TrainerConfigDict) -> Tuple[ModelV2, ActionDistribution]:
    model = build_maddpg_models(policy, obs_space, action_space, config)

    if isinstance(action_space, Simplex):
        return model, TorchDirichlet
    else:
        return model, TorchDeterministic


def maddpg_centralized_critic_postprocessing(policy,
                                             sample_batch,
                                             other_agent_batches=None,
                                             episode=None):
    pytorch = policy.config["framework"] == "torch"
    n_agents = policy.config["model"]["custom_model_config"]["agent_num"]
    opponent_agents_num = n_agents - 1
    continues = True if policy.action_space.__class__ == Box else False
    # action_dim = policy.action_space.n
    if (pytorch and hasattr(policy, "compute_central_vf")) or \
            (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None
        opponent_batch_list = list(other_agent_batches.values())

        # TODO sample batch size not equal across different batches.
        # here we only provide a solution to force the same length with sample batch
        raw_opponent_batch = [opponent_batch_list[i][1] for i in range(opponent_agents_num)]
        opponent_batch = []
        for one_opponent_batch in raw_opponent_batch:
            if len(one_opponent_batch) == len(sample_batch):
                pass
            else:
                if len(one_opponent_batch) > len(sample_batch):
                    one_opponent_batch = one_opponent_batch.slice(0, len(sample_batch))
                else:  # len(one_opponent_batch) < len(sample_batch):
                    length_dif = len(sample_batch) - len(one_opponent_batch)
                    one_opponent_batch = one_opponent_batch.concat(
                        one_opponent_batch.slice(len(one_opponent_batch) - length_dif, len(one_opponent_batch)))
            opponent_batch.append(one_opponent_batch)

        sample_batch["opponent_obs"] = np.stack([opponent_batch[i]["obs"] for i in range(opponent_agents_num)], 1)
        sample_batch["new_opponent_obs"] = np.stack([opponent_batch[i]["new_obs"] for i in range(opponent_agents_num)],
                                                    1)
        sample_batch["opponent_action"] = np.stack([opponent_batch[i]["actions"] for i in range(opponent_agents_num)],
                                                   1)

    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch["opponent_obs"] = np.zeros(
            (sample_batch["obs"].shape[0], opponent_agents_num, sample_batch["obs"].shape[1]),
            dtype=sample_batch["obs"].dtype)
        sample_batch["new_opponent_obs"] = np.zeros(
            (sample_batch["new_obs"].shape[0], opponent_agents_num, sample_batch["new_obs"].shape[1]),
            dtype=sample_batch["new_obs"].dtype)

        sample_batch["opponent_action"] = np.zeros(
            (sample_batch["actions"].shape[0], opponent_agents_num, sample_batch["actions"].shape[1]),
            dtype=sample_batch["actions"].dtype)
        sample_batch["vf_preds"] = np.zeros_like(
            sample_batch["rewards"], dtype=np.float32)

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

    return sample_batch
