from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, \
    Postprocessing, compute_advantages
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from copy import deepcopy
from ray.rllib.evaluation.postprocessing import adjust_nstep
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.numpy import convert_to_numpy
import numpy as np
import copy

torch, nn = try_import_torch()


def get_dim(a):
    dim = 1
    for i in a:
        dim *= i
    return dim


################################################################################################
class CentralizedValueMixin:

    def __init__(self):
        self.compute_central_vf = self.model.central_value_function


def centralized_critic_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):
    custom_config = policy.config["model"]["custom_model_config"]
    pytorch = custom_config["framework"] == "torch"
    obs_dim = get_dim(custom_config["space_obs"]["obs"].shape)
    algorithm = custom_config["algorithm"]
    opp_action_in_cc = custom_config["opp_action_in_cc"]
    global_state_flag = custom_config["global_state_flag"]
    mask_flag = custom_config["mask_flag"]

    if mask_flag:
        action_mask_dim = custom_config["space_act"].n
    else:
        action_mask_dim = 0

    n_agents = custom_config["num_agents"]
    opponent_agents_num = n_agents - 1

    if (pytorch and hasattr(policy, "compute_central_vf")) or \
            (not pytorch and policy.loss_initialized()):

        if not opp_action_in_cc and global_state_flag:
            sample_batch["state"] = sample_batch['obs'][:, action_mask_dim:]
            sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
                convert_to_torch_tensor(
                    sample_batch["state"], policy.device),
            ).cpu().detach().numpy()
        else:  # need opponent info
            assert other_agent_batches is not None
            opponent_batch_list = list(other_agent_batches.values())
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

            # all other agent obs as state
            # sample_batch["state"] = sample_batch['obs'][:, action_mask_dim:action_mask_dim + obs_dim]
            if global_state_flag:  # include self obs and global state
                sample_batch["state"] = sample_batch['obs'][:, action_mask_dim:]
            else:
                sample_batch["state"] = np.stack(
                    [sample_batch['obs'][:, action_mask_dim:action_mask_dim + obs_dim]] + [
                        opponent_batch[i]["obs"][:, action_mask_dim:action_mask_dim + obs_dim] for i in
                        range(opponent_agents_num)], 1)

            sample_batch["opponent_actions"] = np.stack(
                [opponent_batch[i]["actions"] for i in range(opponent_agents_num)],
                1)

            if algorithm in ["coma"]:
                sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
                    convert_to_torch_tensor(
                        sample_batch["state"], policy.device),
                    convert_to_torch_tensor(
                        sample_batch["opponent_actions"], policy.device) if opp_action_in_cc else None,
                ) \
                    .cpu().detach().numpy()
                sample_batch[SampleBatch.VF_PREDS] = np.take(sample_batch[SampleBatch.VF_PREDS],
                                                             np.expand_dims(sample_batch["actions"], axis=1)).squeeze(
                    axis=1)
            else:
                sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
                    convert_to_torch_tensor(
                        sample_batch["state"], policy.device),
                    convert_to_torch_tensor(
                        sample_batch["opponent_actions"], policy.device),
                ) \
                    .cpu().detach().numpy()

    else:
        # Policy hasn't been initialized yet, use zeros.
        o = sample_batch[SampleBatch.CUR_OBS]
        if global_state_flag:
            sample_batch["state"] = np.zeros((o.shape[0], get_dim(custom_config["space_obs"]["state"].shape) + get_dim(
                custom_config["space_obs"]["obs"].shape)),
                                             dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
        else:
            sample_batch["state"] = np.zeros((o.shape[0], n_agents, obs_dim),
                                             dtype=sample_batch[SampleBatch.CUR_OBS].dtype)

        sample_batch["vf_preds"] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)
        sample_batch["opponent_actions"] = np.stack(
            [np.zeros_like(sample_batch["actions"], dtype=sample_batch["actions"].dtype) for _ in
             range(opponent_agents_num)], axis=1)

        # if algorithm in ["coma"]:
        #     sample_batch[SampleBatch.VF_PREDS] = np.take(sample_batch[SampleBatch.VF_PREDS],
        #                                                  np.expand_dims(sample_batch["actions"], axis=1)).squeeze(
        #         axis=1)

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch


################################################################################################
class MixingValueMixin:

    def __init__(self):
        self.mixing_vf = "mixing"


# get opponent value vf
def value_mixing_postprocessing(policy,
                                sample_batch,
                                other_agent_batches=None,
                                episode=None):
    custom_config = policy.config["model"]["custom_model_config"]
    algorithm = custom_config["algorithm"]
    pytorch = custom_config["framework"] == "torch"
    obs_dim = get_dim(custom_config["space_obs"]["obs"].shape)  # 3d input this is channel dim of obs
    if custom_config["global_state_flag"]:
        state_dim = get_dim(custom_config["space_obs"]["state"].shape)
    else:
        state_dim = None

    if custom_config["mask_flag"]:
        action_mask_dim = custom_config["space_act"].n
    else:
        action_mask_dim = 0
    n_agents = custom_config["num_agents"]
    opponent_agents_num = n_agents - 1

    if (pytorch and hasattr(policy, "mixing_vf")) or \
            (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None
        opponent_batch_list = list(other_agent_batches.values())
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

        if state_dim:
            sample_batch["state"] = sample_batch['obs'][:, action_mask_dim + obs_dim:]
        else:  # all other agent obs as state
            # sample_batch["state"] = sample_batch['obs'][:, action_mask_dim:action_mask_dim + obs_dim]
            sample_batch["state"] = np.stack(
                [sample_batch['obs'][:, action_mask_dim:action_mask_dim + obs_dim]] + [
                    opponent_batch[i]["obs"][:, action_mask_dim:action_mask_dim + obs_dim] for i in
                    range(opponent_agents_num)], 1)

        sample_batch["opponent_vf_preds"] = np.stack(
            [opponent_batch[i]["vf_preds"] for i in range(opponent_agents_num)], 1)

        sample_batch["all_vf_preds"] = np.concatenate(
            (np.expand_dims(sample_batch["vf_preds"], axis=1), sample_batch["opponent_vf_preds"]), axis=1)

        sample_batch["vf_tot"] = convert_to_numpy(policy.model.mixing_value(
            convert_to_torch_tensor(sample_batch["all_vf_preds"], policy.device),
            convert_to_torch_tensor(sample_batch["state"], policy.device)))

    else:
        # Policy hasn't been initialized yet, use zeros.
        o = sample_batch[SampleBatch.CUR_OBS]
        if state_dim:
            sample_batch["state"] = np.zeros((o.shape[0], state_dim),
                                             dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
        else:
            sample_batch["state"] = np.zeros((o.shape[0], n_agents, obs_dim),
                                             dtype=sample_batch[SampleBatch.CUR_OBS].dtype)

        sample_batch["vf_preds"] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)
        sample_batch["opponent_vf_preds"] = np.zeros(
            (sample_batch["vf_preds"].shape[0], opponent_agents_num),
            dtype=sample_batch["obs"].dtype)
        sample_batch["all_vf_preds"] = np.concatenate(
            (np.expand_dims(sample_batch["vf_preds"], axis=1), sample_batch["opponent_vf_preds"]), axis=1)

        sample_batch["vf_tot"] = convert_to_numpy(policy.model.mixing_value(
            convert_to_torch_tensor(sample_batch["all_vf_preds"], policy.device),
            convert_to_torch_tensor(sample_batch["state"], policy.device)))

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        # if isinstance(sample_batch["vf_tot"], float):
        #     print(1)
        last_r = sample_batch["vf_tot"][-1]

    train_batch = compute_advantages_vf_tot(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch


def compute_advantages_vf_tot(rollout: SampleBatch,
                              last_r: float,
                              gamma: float = 0.9,
                              lambda_: float = 1.0,
                              use_gae: bool = True,
                              use_critic: bool = True):
    # save the original vf
    vf_saved = deepcopy(rollout[SampleBatch.VF_PREDS])
    rollout[SampleBatch.VF_PREDS] = rollout["vf_tot"]
    rollout = compute_advantages(
        rollout,
        last_r,
        gamma,
        lambda_,
        use_gae,
        use_critic)
    rollout[SampleBatch.VF_PREDS] = vf_saved

    return rollout


################################################################################################
class CentralizedQValueMixin:

    def __init__(self):
        self.compute_central_q = self.model.get_cc_q_values


def centralized_critic_q(policy: Policy,
                         sample_batch: SampleBatch,
                         other_agent_batches=None,
                         episode=None) -> SampleBatch:
    custom_config = policy.config["model"]["custom_model_config"]
    pytorch = custom_config["framework"] == "torch"
    obs_dim = get_dim(custom_config["space_obs"]["obs"].shape)
    algorithm = custom_config["algorithm"]
    opp_action_in_cc = custom_config["opp_action_in_cc"]
    global_state_flag = custom_config["global_state_flag"]
    mask_flag = custom_config["mask_flag"]

    if mask_flag:
        action_mask_dim = custom_config["space_act"].n
    else:
        action_mask_dim = 0

    n_agents = custom_config["num_agents"]
    opponent_agents_num = n_agents - 1

    if (pytorch and hasattr(policy, "compute_central_q")) or \
            (not pytorch and policy.loss_initialized()):

        if not opp_action_in_cc and global_state_flag:
            sample_batch["state"] = sample_batch['obs'][:, action_mask_dim + obs_dim:]
            sample_batch["new_state"] = sample_batch['new_obs'][:, action_mask_dim + obs_dim:]
            raise ValueError("offpolicy centralized critic without action is illegal")

        else:  # need opponent info
            assert other_agent_batches is not None
            opponent_batch_list = list(other_agent_batches.values())
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

            # all other agent obs as state
            # sample_batch["state"] = sample_batch['obs'][:, action_mask_dim:action_mask_dim + obs_dim]
            if global_state_flag:
                sample_batch["state"] = sample_batch['obs'][:, action_mask_dim + obs_dim:]
                sample_batch["new_state"] = sample_batch['new_obs'][:, action_mask_dim + obs_dim:]

            else:
                sample_batch["state"] = np.stack(
                    [sample_batch['obs'][:, action_mask_dim:action_mask_dim + obs_dim]] + [
                        opponent_batch[i]["obs"][:, action_mask_dim:action_mask_dim + obs_dim] for i in
                        range(opponent_agents_num)], 1)
                sample_batch["new_state"] = np.stack(
                    [sample_batch['new_obs'][:, action_mask_dim:action_mask_dim + obs_dim]] + [
                        opponent_batch[i]["new_obs"][:, action_mask_dim:action_mask_dim + obs_dim] for i in
                        range(opponent_agents_num)], 1)

            sample_batch["opponent_actions"] = np.stack(
                [opponent_batch[i]["actions"] for i in range(opponent_agents_num)],
                1)
            sample_batch["prev_opponent_actions"] = np.stack(
                [opponent_batch[i]["prev_actions"] for i in range(opponent_agents_num)],
                1)

            # grab the opponent next action manually
            all_opponent_batch_next_action_ls = []
            for opp_index in range(opponent_agents_num):
                opp_policy = opponent_batch_list[opp_index][0]
                opp_batch = copy.deepcopy(opponent_batch[opp_index])
                input_dict = {}
                input_dict["obs"] = {}
                input_dict["obs"]["obs"] = opp_batch["new_obs"][:,
                                           action_mask_dim: action_mask_dim + obs_dim]
                seq_lens = opp_batch["seq_lens"]
                state_ls = []
                start_point = 0
                for seq_len in seq_lens:
                    state = convert_to_torch_tensor(opp_batch["state_out_0"][start_point], policy.device)
                    start_point += seq_len
                    state_ls.append(state)
                state = [torch.stack(state_ls, 0)]
                input_dict = convert_to_torch_tensor(input_dict, policy.device)
                seq_lens = convert_to_torch_tensor(seq_lens, policy.device)
                opp_next_action, _ = opp_policy.model.policy_model(input_dict, state, seq_lens)
                opp_next_action = convert_to_numpy(opp_next_action)
                all_opponent_batch_next_action_ls.append(opp_next_action)
            sample_batch["next_opponent_actions"] = np.stack(
                all_opponent_batch_next_action_ls, 1)

    else:
        # Policy hasn't been initialized yet, use zeros.
        o = sample_batch[SampleBatch.CUR_OBS]
        if global_state_flag:
            sample_batch["state"] = np.zeros((o.shape[0], get_dim(custom_config["space_obs"]["state"].shape)),
                                             dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
            sample_batch["new_state"] = np.zeros((o.shape[0], get_dim(custom_config["space_obs"]["state"].shape)),
                                                 dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
        else:
            sample_batch["state"] = np.zeros((o.shape[0], n_agents, obs_dim),
                                             dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
            sample_batch["new_state"] = np.zeros((o.shape[0], n_agents, obs_dim),
                                                 dtype=sample_batch[SampleBatch.CUR_OBS].dtype)

        sample_batch["vf_preds"] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)
        sample_batch["opponent_actions"] = np.stack(
            [np.zeros_like(sample_batch["actions"], dtype=sample_batch["actions"].dtype) for _ in
             range(opponent_agents_num)], axis=1)
        sample_batch["prev_opponent_actions"] = np.stack(
            [np.zeros_like(sample_batch["actions"], dtype=sample_batch["actions"].dtype) for _ in
             range(opponent_agents_num)], axis=1)
        sample_batch["next_opponent_actions"] = np.stack(
            [np.zeros_like(sample_batch["actions"], dtype=sample_batch["actions"].dtype) for _ in
             range(opponent_agents_num)], axis=1)

    # N-step Q adjustments.
    if policy.config["n_step"] > 1:
        adjust_nstep(policy.config["n_step"], policy.config["gamma"], sample_batch)

    # Create dummy prio-weights (1.0) in case we don't have any in
    # the batch.
    if "weights" not in sample_batch:
        sample_batch["weights"] = np.ones_like(sample_batch[SampleBatch.REWARDS])

    # Prioritize on the worker side.
    if sample_batch.count > 0 and policy.config["worker_side_prioritization"]:
        td_errors = policy.compute_td_error(
            sample_batch[SampleBatch.OBS], sample_batch[SampleBatch.ACTIONS],
            sample_batch[SampleBatch.REWARDS], sample_batch[SampleBatch.NEXT_OBS],
            sample_batch[SampleBatch.DONES], sample_batch["weights"])
        new_priorities = (np.abs(convert_to_numpy(td_errors)) +
                          policy.config["prioritized_replay_eps"])
        sample_batch["weights"] = new_priorities

    return sample_batch


################################################################################################
class MixingQValueMixin:

    def __init__(self):
        self.compute_mixing_q = True


def q_value_mixing(policy: Policy,
                   sample_batch: SampleBatch,
                   other_agent_batches=None,
                   episode=None) -> SampleBatch:
    custom_config = policy.config["model"]["custom_model_config"]
    pytorch = custom_config["framework"] == "torch"
    obs_dim = get_dim(custom_config["space_obs"]["obs"].shape)
    algorithm = custom_config["algorithm"]
    opp_action_in_cc = custom_config["opp_action_in_cc"]
    global_state_flag = custom_config["global_state_flag"]
    mask_flag = custom_config["mask_flag"]

    if mask_flag:
        action_mask_dim = custom_config["space_act"].n
    else:
        action_mask_dim = 0

    n_agents = custom_config["num_agents"]
    opponent_agents_num = n_agents - 1

    if (pytorch and hasattr(policy, "compute_mixing_q")) or \
            (not pytorch and policy.loss_initialized()):

        if not opp_action_in_cc and global_state_flag:
            sample_batch["state"] = sample_batch['obs'][:, action_mask_dim + obs_dim:]
            sample_batch["new_state"] = sample_batch['new_obs'][:, action_mask_dim + obs_dim:]

        else:  # need opponent info
            assert other_agent_batches is not None
            opponent_batch_list = list(other_agent_batches.values())
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

            # all other agent obs as state
            # sample_batch["state"] = sample_batch['obs'][:, action_mask_dim:action_mask_dim + obs_dim]
            if global_state_flag:
                sample_batch["state"] = sample_batch['obs'][:, action_mask_dim + obs_dim:]
                sample_batch["new_state"] = sample_batch['new_obs'][:, action_mask_dim + obs_dim:]

            else:
                sample_batch["state"] = np.stack(
                    [sample_batch['obs'][:, action_mask_dim:action_mask_dim + obs_dim]] + [
                        opponent_batch[i]["obs"][:, action_mask_dim:action_mask_dim + obs_dim] for i in
                        range(opponent_agents_num)], 1)
                sample_batch["new_state"] = np.stack(
                    [sample_batch['new_obs'][:, action_mask_dim:action_mask_dim + obs_dim]] + [
                        opponent_batch[i]["new_obs"][:, action_mask_dim:action_mask_dim + obs_dim] for i in
                        range(opponent_agents_num)], 1)

            # grab the opponent Q
            all_opponent_batch_q_ls = []
            for opp_index in range(opponent_agents_num):
                opp_policy = opponent_batch_list[opp_index][0]
                opp_batch = copy.deepcopy(opponent_batch[opp_index])
                input_dict = {}
                input_dict["obs"] = {}
                input_dict["obs"]["obs"] = opp_batch["obs"][:,
                                           action_mask_dim: action_mask_dim + obs_dim]
                input_dict["actions"] = opp_batch["actions"]
                seq_lens = opp_batch["seq_lens"]
                state_ls = []
                for i, seq_len in enumerate(seq_lens):
                    state = convert_to_torch_tensor(opp_batch["state_in_1"][i], policy.device)
                    state_ls.append(state)
                state = [torch.stack(state_ls, 0)]
                input_dict = convert_to_torch_tensor(input_dict, policy.device)
                seq_lens = convert_to_torch_tensor(seq_lens, policy.device)
                opp_q, _ = opp_policy.model.q_model(input_dict, state, seq_lens)
                opp_q = convert_to_numpy(opp_q.squeeze(1))
                all_opponent_batch_q_ls.append(opp_q)
            sample_batch["opponent_q"] = np.stack(
                all_opponent_batch_q_ls, 1)

            # grab the opponent next action & compute next Q use target net
            all_opponent_batch_next_q_ls = []
            for opp_index in range(opponent_agents_num):
                opp_policy = opponent_batch_list[opp_index][0]
                opp_batch = copy.deepcopy(opponent_batch[opp_index])
                input_dict = {}
                input_dict["obs"] = {}
                input_dict["obs"]["obs"] = opp_batch["new_obs"][:,
                                           action_mask_dim: action_mask_dim + obs_dim]
                seq_lens_array = opp_batch["seq_lens"]
                seq_lens = convert_to_torch_tensor(seq_lens_array, policy.device)

                state_ls = []
                start_point = 0
                for seq_len in seq_lens_array:
                    state = convert_to_torch_tensor(opp_batch["state_out_0"][start_point], policy.device)
                    start_point += seq_len
                    state_ls.append(state)
                state = [torch.stack(state_ls, 0)]

                input_dict = convert_to_torch_tensor(input_dict, policy.device)
                opp_next_action, _ = opp_policy.target_model.policy_model(input_dict, state, seq_lens)

                state_ls = []
                start_point = 0
                for seq_len in seq_lens_array:
                    state = convert_to_torch_tensor(opp_batch["state_out_1"][start_point], policy.device)
                    start_point += seq_len
                    state_ls.append(state)
                state = [torch.stack(state_ls, 0)]

                input_dict["actions"] = opp_next_action
                next_opp_q, _ = opp_policy.target_model.q_model(input_dict, state, seq_lens)

                next_opp_q = convert_to_numpy(next_opp_q.squeeze(1))
                all_opponent_batch_next_q_ls.append(next_opp_q)
            sample_batch["next_opponent_q"] = np.stack(
                all_opponent_batch_next_q_ls, 1)

    else:
        # Policy hasn't been initialized yet, use zeros.
        o = sample_batch[SampleBatch.CUR_OBS]
        if global_state_flag:
            sample_batch["state"] = np.zeros((o.shape[0], get_dim(custom_config["space_obs"]["state"].shape)),
                                             dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
            sample_batch["new_state"] = np.zeros((o.shape[0], get_dim(custom_config["space_obs"]["state"].shape)),
                                                 dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
        else:
            sample_batch["state"] = np.zeros((o.shape[0], n_agents, obs_dim),
                                             dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
            sample_batch["new_state"] = np.zeros((o.shape[0], n_agents, obs_dim),
                                                 dtype=sample_batch[SampleBatch.CUR_OBS].dtype)

        sample_batch["opponent_q"] = np.zeros(
            (sample_batch["state"].shape[0], opponent_agents_num),
            dtype=sample_batch["obs"].dtype)
        sample_batch["next_opponent_q"] = np.zeros(
            (sample_batch["state"].shape[0], opponent_agents_num),
            dtype=sample_batch["obs"].dtype)

    # N-step Q adjustments.
    if policy.config["n_step"] > 1:
        adjust_nstep(policy.config["n_step"], policy.config["gamma"], sample_batch)

    # Create dummy prio-weights (1.0) in case we don't have any in
    # the batch.
    if "weights" not in sample_batch:
        sample_batch["weights"] = np.ones_like(sample_batch[SampleBatch.REWARDS])

    # Prioritize on the worker side.
    if sample_batch.count > 0 and policy.config["worker_side_prioritization"]:
        td_errors = policy.compute_td_error(
            sample_batch[SampleBatch.OBS], sample_batch[SampleBatch.ACTIONS],
            sample_batch[SampleBatch.REWARDS], sample_batch[SampleBatch.NEXT_OBS],
            sample_batch[SampleBatch.DONES], sample_batch["weights"])
        new_priorities = (np.abs(convert_to_numpy(td_errors)) +
                          policy.config["prioritized_replay_eps"])
        sample_batch["weights"] = new_priorities

    return sample_batch
