from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
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

##############
# FACMAC
##############
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
