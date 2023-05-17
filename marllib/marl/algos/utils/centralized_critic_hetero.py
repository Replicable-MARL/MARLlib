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

import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import discount_cumsum, Postprocessing, compute_gae_for_sample_batch
from marllib.marl.algos.utils.centralized_critic import convert_to_torch_tensor
from marllib.marl.algos.utils.setup_utils import get_agent_num
from marllib.marl.algos.utils.centralized_Q import get_dim
from ray.rllib.utils.framework import try_import_torch
from marllib.marl.algos.utils.mixing_Q import align_batch

torch, nn = try_import_torch()

"""
centralized critic postprocessing for 
1. HAPPO 
2. HATRPO 
"""

GLOBAL_NEED_COLLECT = [SampleBatch.ACTION_LOGP, SampleBatch.ACTIONS,
                       SampleBatch.ACTION_DIST_INPUTS, SampleBatch.OBS]
GLOBAL_PREFIX = 'global_'
GLOBAL_MODEL_LOGITS = f'{GLOBAL_PREFIX}model_logits'
GLOBAL_IS_TRAINING = f'{GLOBAL_PREFIX}is_trainable'
GLOBAL_TRAIN_BATCH = f'{GLOBAL_PREFIX}train_batch'
STATE = 'state'
MODEL = 'model'
POLICY_ID = 'policy_id'
TRAINING = 'training'


def get_global_name(key, i=None):
    if i == 'self': return key
    # converts a key to global format

    return f'{GLOBAL_PREFIX}{key}_agent_{i}'


def add_all_agents_gae(policy, sample_batch, other_agent_batches=None, episode=None):
    # print('------------step into post processing ----------\n'*8)
    sample_batch = add_opponent_information_and_critical_vf(policy, sample_batch, other_agent_batches, episode=episode)
    train_batch = compute_gae_for_sample_batch(policy, sample_batch, other_agent_batches, episode)

    return train_batch


Postprocessing.DELTAS = 'DELTA'
Postprocessing.RETURNS = 'RETURNS'


def _get_last_r(policy, sample_batch):
    if sample_batch[SampleBatch.DONES][-1]:
        last_r = 0.0
    # Trajectory has been truncated -> last r=VF estimate of last obs.
    else:
        # Input dict is provided to us automatically via the Model's
        # requirements. It's a single-timestep (last one in trajectory)
        # input_dict.
        # Create an input dict according to the Model's requirements.
        input_dict = sample_batch.get_single_step_input_dict(
            policy.model.view_requirements, index="last")
        last_r = policy._value(**input_dict)

    return last_r


def _add_deltas(sample_batch, last_r, gamma):
    vpred_t = np.concatenate(
        [sample_batch[SampleBatch.VF_PREDS],
         np.array([last_r])]
    )
    delta_t = (
            sample_batch[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1]
    )

    sample_batch[Postprocessing.DELTAS] = delta_t

    return sample_batch


def _add_returns(sample_batch, last_r, gamma):
    rewards_plus_v = np.concatenate(
        [sample_batch[SampleBatch.REWARDS],
         np.array([last_r])])
    discounted_returns = discount_cumsum(rewards_plus_v,
                                         gamma)[:-1].astype(np.float32)

    sample_batch[Postprocessing.RETURNS] = discounted_returns

    return sample_batch


def trpo_post_process(policy, sample_batch, other_agent_batches=None, episode=None):
    sample_batch = compute_gae_for_sample_batch(policy, sample_batch, other_agent_batches, episode)

    last_r = _get_last_r(policy, sample_batch)
    gamma = policy.config["gamma"]

    sample_batch = _add_returns(sample_batch=sample_batch, last_r=last_r, gamma=gamma)
    sample_batch = _add_deltas(sample_batch=sample_batch, last_r=last_r, gamma=gamma)

    return sample_batch


def hatrpo_post_process(policy, sample_batch, other_agent_batches=None, episode=None):
    sample_batch = add_all_agents_gae(policy, sample_batch, other_agent_batches, episode)

    last_r = _get_last_r(policy, sample_batch)
    gamma = policy.config["gamma"]

    sample_batch = _add_returns(sample_batch=sample_batch, last_r=last_r, gamma=gamma)
    sample_batch = _add_deltas(sample_batch=sample_batch, last_r=last_r, gamma=gamma)

    n_agents = get_agent_num(policy)

    if other_agent_batches:
        for name in other_agent_batches:
            _p, _b = other_agent_batches[name]
            cur_training = _b.is_training

            sample_batch[get_global_name(TRAINING, name)] = np.array([
                                                                         int(cur_training)
                                                                     ] * len(sample_batch))

    return sample_batch


def contain_global_obs(train_batch):
    return any(key.startswith(GLOBAL_PREFIX) for key in train_batch)


def add_other_agent_mul_info(sample_batch, other_agent_info, agent_num):
    global GLOBAL_NEED_COLLECT

    if other_agent_info:
        for name in other_agent_info:
            for key in GLOBAL_NEED_COLLECT:
                if key not in sample_batch: continue

                _p, _b = other_agent_info[name]  # _p means policy, _b means batch
                sample_batch[get_global_name(key, name)] = _b[key]

    return sample_batch


def get_vf_pred(policy, algorithm, sample_batch, opp_action_in_cc):
    sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
        convert_to_torch_tensor(
            sample_batch[STATE], policy.device),
        convert_to_torch_tensor(
            sample_batch['opponent_actions'], policy.device) if opp_action_in_cc else None,
    ) \
        .cpu().detach().numpy()

    return sample_batch


def link_with_other_agents(current_policy, agent_num, sample_batches, other_agent_info):
    # ic('self')
    # ic(torch.std(flat_params(current_policy.model.actor_parameters())))
    # ic(torch.mean(flat_params(current_policy.model.actor_parameters())))
    if not other_agent_info:
        pass
    else:
        policies = [current_policy]

        # set all the other-models critic as current policy critic

        for name in other_agent_info:
            _p, _b = other_agent_info[name]

            if any([
                _p.model.cc_vf_encoder is not current_policy.model.cc_vf_encoder,
                _p.model.cc_vf_branch is not current_policy.model.cc_vf_branch,
                _p.model.vf_encoder is not current_policy.model.vf_encoder,
                _p.model.vf_branch is not current_policy.model.vf_branch]):
                _p.model.cc_vf_encoder = current_policy.model.cc_vf_encoder
                _p.model.cc_vf_branch = current_policy.model.cc_vf_branch
                _p.model.vf_encoder = current_policy.model.vf_encoder
                _p.model.vf_branch = current_policy.model.vf_branch

            current_policy.model.link_other_agent_policy(name, _p)

def collect_opponent_array(other_agent_batches: dict, opponent_agents_num, sample_batch):
    assert other_agent_batches is not None
    opponent_batch_list = list(other_agent_batches.values())
    raw_opponent_batch = [opponent_batch_list[i][1] for i in range(opponent_agents_num)]

    opponent_batch = []

    for i, one_opponent_batch in enumerate(raw_opponent_batch):
        one_opponent_batch = align_batch(one_opponent_batch, sample_batch)
        opponent_batch.append(one_opponent_batch)

    return opponent_batch


def state_name(i): return f'state_in_{i}'


def global_state_name(i, ai): return f'state_in_{i}_of_agent_{ai}'


def add_state_in_for_opponent(sample_batch, other_agent_batches, agent_num):
    state_in_num = 0

    while state_name(state_in_num) in sample_batch:
        state_in_num += 1

    # get how many state in layer
    # agent_indices = [0, 1, 2]

    if other_agent_batches:
        for name in other_agent_batches:
            # name = agent(a_i)

            for s_i in range(state_in_num):
                opponent_state_exist = False
                if other_agent_batches:
                    if state_name(s_i) in other_agent_batches[name]:
                        _policy, _batch = other_agent_batches[name]
                        sample_batch[global_state_name(s_i, name)] = _batch[state_name(s_i)]
                        opponent_state_exist = True

                if not opponent_state_exist:
                    sample_batch[global_state_name(s_i, name)] = np.zeros_like(
                        sample_batch[state_name(s_i)],
                        dtype=sample_batch[state_name(s_i)].dtype
                    )

    return sample_batch


def add_opponent_information_and_critical_vf(policy,
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

    n_agents = get_agent_num(policy)
    opponent_agents_num = n_agents - 1

    opponent_info_exists = (pytorch and hasattr(policy, "compute_central_vf")) or (
            not pytorch and policy.loss_initialized())

    if opponent_info_exists:
        if not opp_action_in_cc and global_state_flag:
            sample_batch[STATE] = sample_batch[SampleBatch.OBS][:, action_mask_dim:]
            sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
                convert_to_torch_tensor(
                    sample_batch[STATE], policy.device),
            ).cpu().detach().numpy()
        else:  # need opponent info
            opponent_batch = collect_opponent_array(other_agent_batches=other_agent_batches,
                                                    opponent_agents_num=opponent_agents_num,
                                                    sample_batch=sample_batch)

            # all other agent obs as state
            if global_state_flag:  # include self obs and global state
                sample_batch[STATE] = sample_batch[SampleBatch.OBS][:, action_mask_dim:]
            else:
                sample_batch[STATE] = np.stack(
                    [sample_batch[SampleBatch.OBS][:, action_mask_dim:action_mask_dim + obs_dim]] + [
                        opponent_batch[i][SampleBatch.OBS][:, action_mask_dim:action_mask_dim + obs_dim] for i in
                        range(opponent_agents_num)], 1)

            sample_batch["opponent_actions"] = np.stack(
                [opponent_batch[i]["actions"] for i in range(opponent_agents_num)],
                1)
    else:
        # Policy hasn't been initialized yet, use zeros.
        o = sample_batch[SampleBatch.CUR_OBS]
        if global_state_flag:
            sample_batch[STATE] = np.zeros((o.shape[0], get_dim(custom_config["space_obs"]["state"].shape) + get_dim(
                custom_config["space_obs"]["obs"].shape)),
                                           dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
        else:
            sample_batch[STATE] = np.zeros((o.shape[0], n_agents, obs_dim),
                                           dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
        sample_batch["opponent_actions"] = np.stack(
            [np.zeros_like(sample_batch["actions"], dtype=sample_batch["actions"].dtype) for _ in
             range(opponent_agents_num)], axis=1)

    if algorithm.upper() in ['HAPPO', 'HATRPO']:
        link_with_other_agents(policy, n_agents, sample_batch, other_agent_batches)

    sample_batch = add_other_agent_mul_info(
        sample_batch=sample_batch,
        other_agent_info=other_agent_batches,
        agent_num=n_agents,
    )

    sample_batch = add_state_in_for_opponent(
        sample_batch=sample_batch,
        other_agent_batches=other_agent_batches,
        agent_num=n_agents,
    )

    if opponent_info_exists:
        sample_batch = get_vf_pred(policy, algorithm, sample_batch, opp_action_in_cc)
    else:
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    return sample_batch
