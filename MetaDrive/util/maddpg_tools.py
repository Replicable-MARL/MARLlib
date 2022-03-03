from ray.rllib.agents.ddpg.noop_model import NoopModel, TorchNoopModel
from ray.rllib.models.catalog import ModelCatalog

from ray.rllib.utils.tf_ops import huber_loss, make_tf_callable
import gym
from typing import Dict, Tuple

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
from MetaDrive.model.torch_maddpg import MADDPGTorchModel
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable
from ray.rllib.evaluation.postprocessing import adjust_nstep
from ray.rllib.utils.numpy import convert_to_numpy
import numpy as np
from ray.rllib.utils.framework import try_import_tf, try_import_torch

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
    cc_dim = policy.config["centralized_critic_obs_dim"]

    if policy.config["fuse_mode"] == "mf":

        cc_input_dict = {
            "obs": train_batch['centralized_critic_obs'][:, cc_dim // 2:],  # latter half is the cc input
            "is_training": True,
        }

        cc_input_dict_next = {
            "obs": train_batch['new_centralized_critic_obs'][:, cc_dim // 2:],
            "is_training": True,
        }

        cc_model_out_t, _ = model(cc_input_dict, [], None)
        cc_target_model_out_tp1, _ = target_model(cc_input_dict_next, [], None)

        cc_model_out_t_ls = [cc_model_out_t]
        cc_target_model_out_tp1_ls = [cc_target_model_out_tp1]

    else:  # concat mode
        opp_num = policy.config["num_neighbours"]
        cc_model_out_t_ls = []
        cc_target_model_out_tp1_ls = []
        for index in range(1, opp_num + 1):
            indi_obs_dim = cc_dim // (opp_num + 1)
            cc_input_dict = {
                "obs": train_batch['centralized_critic_obs'][:, index * indi_obs_dim:index * indi_obs_dim + indi_obs_dim],
                # latter half is the cc input
                "is_training": True,
            }
            cc_input_dict_next = {
                "obs": train_batch['new_centralized_critic_obs'][:, index * indi_obs_dim:index * indi_obs_dim + indi_obs_dim],
                "is_training": True,
            }

            cc_model_out_t, _ = model(cc_input_dict, [], None)
            cc_target_model_out_tp1, _ = target_model(cc_input_dict_next, [], None)

            cc_model_out_t_ls.append(cc_model_out_t)
            cc_target_model_out_tp1_ls.append(cc_target_model_out_tp1)

    model_out_t, _ = model(input_dict, [], None)

    # model_out_tp1, _ = model(input_dict_next, [], None)
    target_model_out_tp1, _ = target_model(input_dict_next, [], None)

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
    q_t = model.get_q_values(model_out_t, cc_model_out_t_ls, train_batch[SampleBatch.ACTIONS])

    # Q-values for current policy (no noise) in given current state
    q_t_det_policy = model.get_q_values(model_out_t, cc_model_out_t_ls, policy_t)

    actor_loss = -torch.mean(q_t_det_policy)

    if twin_q:
        twin_q_t = model.get_twin_q_values(model_out_t,
                                           train_batch[SampleBatch.ACTIONS])
    # q_batchnorm_update_ops = list(
    #     set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS)) - prev_update_ops)

    # Target q-net(s) evaluation.
    # here target_model_out_tp1 + target_opp_model_out_t_ls ascentralized critic
    q_tp1 = target_model.get_q_values(target_model_out_tp1, cc_target_model_out_tp1_ls,
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
    if policy.config["use_state_preprocessor"]:
        default_model = None  # catalog decides
        num_outputs = 256  # arbitrary
        config["model"]["no_final_linear"] = True
    else:
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


def maddpg_centralized_critic_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):
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

        sample_batch["new_centralized_critic_obs"] = np.zeros(
            (o.shape[0], policy.config["centralized_critic_obs_dim"]), dtype=sample_batch[SampleBatch.NEXT_OBS].dtype
        )
        sample_batch["new_centralized_critic_obs"][:, :odim] = sample_batch[SampleBatch.NEXT_OBS]

        if policy.config["fuse_mode"] == "concat":
            sample_batch = maddpg_concat_cc_process(policy, sample_batch, other_agent_batches, odim, adim,
                                                    other_info_dim)
        elif policy.config["fuse_mode"] == "mf":
            sample_batch = maddpg_mean_field_cc_process(
                policy, sample_batch, other_agent_batches, odim, adim, other_info_dim
            )
        else:
            raise ValueError("Unknown fuse mode: {}".format(policy.config["fuse_mode"]))

    else:
        # Policy hasn't been initialized yet, use zeros.
        _ = sample_batch[SampleBatch.INFOS]  # touch
        o = sample_batch[SampleBatch.CUR_OBS]
        sample_batch["centralized_critic_obs"] = np.zeros(
            (o.shape[0], policy.config["centralized_critic_obs_dim"]), dtype=o.dtype
        )
        sample_batch["new_centralized_critic_obs"] = np.zeros(
            (o.shape[0], policy.config["centralized_critic_obs_dim"]), dtype=o.dtype
        )

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


def maddpg_concat_cc_process(policy, sample_batch, other_agent_batches, odim, adim, other_info_dim):
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
                    sample_batch["new_centralized_critic_obs"][index, n_count * odim: (n_count + 1) * odim] = \
                        other_agent_batches[n_name][1][SampleBatch.NEXT_OBS][index]
    return sample_batch


def maddpg_mean_field_cc_process(policy, sample_batch, other_agent_batches, odim, adim, other_info_dim):
    """Average the neighbors' observations"""
    for index in range(sample_batch.count):
        neighbours = sample_batch['infos'][index]["neighbours"]
        neighbours_distance = sample_batch['infos'][index]["neighbours_distance"]
        obs_list = []
        new_obs_list = []
        act_list = []
        for n_count, (n_name, n_dist) in enumerate(zip(neighbours, neighbours_distance)):
            if n_dist > policy.config["mf_nei_distance"]:
                continue
            if n_name in other_agent_batches and \
                    (index < len(other_agent_batches[n_name][1][SampleBatch.CUR_OBS])):
                obs_list.append(other_agent_batches[n_name][1][SampleBatch.CUR_OBS][index])
                new_obs_list.append(other_agent_batches[n_name][1][SampleBatch.NEXT_OBS][index])
                act_list.append(other_agent_batches[n_name][1][SampleBatch.ACTIONS][index])
        if len(obs_list) > 0:
            sample_batch["centralized_critic_obs"][index, odim:odim + odim] = np.mean(obs_list, axis=0)
            sample_batch["new_centralized_critic_obs"][index, odim:odim + odim] = np.mean(new_obs_list, axis=0)

            if policy.config["counterfactual"]:
                sample_batch["centralized_critic_obs"][index, odim + odim:odim + odim + adim] = np.mean(act_list,
                                                                                                        axis=0)
    return sample_batch
