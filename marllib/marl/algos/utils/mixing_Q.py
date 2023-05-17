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

from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.evaluation.postprocessing import adjust_nstep
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.numpy import convert_to_numpy
from marllib.marl.algos.utils.centralized_Q import get_dim
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
import copy
import numpy as np

torch, nn = try_import_torch()

"""
mixing Q postprocessing for 
1. FACMAC 
"""


class MixingQValueMixin:

    def __init__(self):
        self.compute_mixing_q = True


def align_batch(one_opponent_batch, sample_batch):
    if len(one_opponent_batch) == len(sample_batch):
        pass
    else:
        if len(one_opponent_batch) > len(sample_batch):
            one_opponent_batch = one_opponent_batch.slice(0, len(sample_batch))
        else:  # len(one_opponent_batch) < len(sample_batch):
            length_dif = len(sample_batch) - len(one_opponent_batch)
            one_opponent_batch = one_opponent_batch.concat(
                one_opponent_batch.slice(len(one_opponent_batch) - length_dif, len(one_opponent_batch)))
    return one_opponent_batch


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
                one_opponent_batch = align_batch(one_opponent_batch, sample_batch)
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

    # N-step rewards adjustments.
    if policy.config["n_step"] > 1:
        adjust_nstep(policy.config["n_step"], policy.config["gamma"], sample_batch)

    # Create dummy prio-weights (1.0) in case we don't have any in
    # the batch.
    if "weights" not in sample_batch:
        sample_batch["weights"] = np.ones_like(sample_batch[SampleBatch.REWARDS])

    return sample_batch


# postprocessing sampled batch before learning stage.
def before_learn_on_batch(multi_agent_batch, policies, train_batch_size):
    all_agent_q = []
    all_agent_target_q = []
    for pid, policy in policies.items():

        custom_config = policy.config["model"]["custom_model_config"]
        obs_dim = get_dim(custom_config["space_obs"]["obs"].shape)
        global_state_flag = custom_config["global_state_flag"]
        n_agents = custom_config["num_agents"]

        policy_batch = copy.deepcopy(multi_agent_batch.policy_batches[pid])
        policy_batch["agent_index"] = policy_batch["agent_index"] + 1
        pad_batch_to_sequences_of_same_size(
            batch=policy_batch,
            max_seq_len=policy.max_seq_len,
            shuffle=False,
            batch_divisibility_req=policy.batch_divisibility_req,
            view_requirements=policy.view_requirements,
        )

        q_model = policy.model.q_model.to(policy.device)
        target_policy_model = policy.target_model.policy_model.to(policy.device)
        target_q_model = policy.target_model.q_model.to(policy.device)

        obs = policy_batch["obs"]
        next_obs = policy_batch["new_obs"]
        if "state_in_2" not in policy_batch:
            state_in_p = [policy_batch["state_in_0"]]
            state_in_q = [policy_batch["state_in_1"]]
        else:
            state_in_p = [policy_batch["state_in_0"], policy_batch["state_in_1"]]
            state_in_q = [policy_batch["state_in_2"], policy_batch["state_in_3"]]
        seq_lens = policy_batch["seq_lens"]

        input_dict = {"obs": {}}
        target_input_dict = {"obs": {}}

        input_dict["obs"]["obs"] = obs
        target_input_dict["obs"]["obs"] = next_obs

        if global_state_flag:
            input_dict["obs"]["obs"] = obs[:, :obs_dim]
            input_dict["state"] = obs[:, obs_dim:]
            target_input_dict["obs"]["obs"] = next_obs[:, :obs_dim]
            target_input_dict["state"] = next_obs[:, obs_dim:]

        # get current action & Q value
        action = policy_batch["actions"]
        input_dict["actions"] = action
        seq_lens = convert_to_torch_tensor(seq_lens, policy.device)

        input_dict = convert_to_torch_tensor(input_dict, policy.device)
        state_in_q = [convert_to_torch_tensor(state_rnn, policy.device) for state_rnn in state_in_q]
        q, _ = q_model.forward(input_dict, state_in_q, seq_lens)
        q = convert_to_numpy(q)

        # get next action & Q value from target model
        target_input_dict = convert_to_torch_tensor(target_input_dict, policy.device)
        state_in_p = [convert_to_torch_tensor(state_rnn, policy.device) for state_rnn in state_in_p]
        next_action_out, _ = target_policy_model.forward(target_input_dict, state_in_p, seq_lens)
        next_action = target_policy_model.action_out_squashed(next_action_out)
        target_input_dict["actions"] = next_action

        next_target_q, _ = target_q_model.forward(target_input_dict, state_in_q, seq_lens)
        next_target_q = convert_to_numpy(next_target_q)

        agent_id = np.unique(policy_batch["agent_index"])
        for a_id in agent_id:
            if a_id == 0:  # zero padding
                continue
            valid_flag = np.where(policy_batch["agent_index"] == a_id)[0]
            q_one_agent = q[valid_flag, :]
            next_target_q_one_agent = next_target_q[valid_flag, :]

            all_agent_q.append(q_one_agent)
            all_agent_target_q.append(next_target_q_one_agent)

    # construct opponent q for each batch
    all_agent_q = np.stack(all_agent_q, 1)
    all_agent_target_q = np.stack(all_agent_target_q, 1)

    for pid, policy in policies.items():
        policy_batch = multi_agent_batch.policy_batches[pid]
        agent_id = np.unique(policy_batch["agent_index"])
        agent_num = len(agent_id)
        all_agent_q_ls = []
        all_agent_target_q_ls = []

        for a in range(agent_num):
            all_agent_q_ls.append(all_agent_q)
            all_agent_target_q_ls.append(all_agent_target_q)

        q_batch = np.stack(all_agent_q_ls, 1).reshape((policy_batch.count, -1))
        target_q_batch = np.stack(all_agent_target_q_ls, 1).reshape((policy_batch.count, -1))

        other_q_batch_ls = []
        other_target_q_batch_ls = []

        for i in range(policy_batch.count):
            current_agent_id = policy_batch["agent_index"][i]
            q_ts = q_batch[i]
            target_q_ts = target_q_batch[i]

            other_q_ts = np.delete(q_ts, current_agent_id, axis=0)
            other_target_q_ts = np.delete(target_q_ts, current_agent_id, axis=0)

            other_q_batch_ls.append(other_q_ts)
            other_target_q_batch_ls.append(other_target_q_ts)

        other_q_batch = np.stack(other_q_batch_ls, 0)
        other_target_q_batch = np.stack(other_target_q_batch_ls, 0)

        multi_agent_batch.policy_batches[pid]["opponent_q"] = other_q_batch
        multi_agent_batch.policy_batches[pid]["next_opponent_q"] = other_target_q_batch

    return multi_agent_batch
