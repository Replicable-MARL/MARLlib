from ray.rllib.utils.framework import try_import_torch
from ray.rllib.evaluation.postprocessing import adjust_nstep
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.numpy import convert_to_numpy
import numpy as np

torch, nn = try_import_torch()


def get_dim(a):
    dim = 1
    for i in a:
        dim *= i
    return dim


##############
# MADDPG
##############
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
