from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np
from ray.rllib.evaluation.postprocessing import discount_cumsum, Postprocessing, compute_gae_for_sample_batch
from marl.algos.utils.valuenorm import ValueNorm
from marl.algos.utils.centralized_critic import convert_to_torch_tensor
from marl.algos.utils.setup_utils import get_agent_num
from marl.algos.utils.centralized_Q import get_dim
import multiprocessing

"""
centralized critic postprocessing for 
1. HAPPO 
2. HATRPO 
"""

mul_manager = multiprocessing.Manager()

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

value_normalizer = ValueNorm(1)


def get_global_name(key, i=None):
    # converts a key to global format

    if i is None: i = 'all'

    return f'{GLOBAL_PREFIX}{key}_agent_{i}'


def _extract_from_all_others(take_fn):
    def inner(other_agent_batches):
        agent_ids = sorted([agent_id for agent_id in other_agent_batches])

        return [
            take_fn(other_agent_batches[_id]) for _id in agent_ids
        ]

    return inner


def extract_other_agents_train_batch(other_agent_batches):
    return _extract_from_all_others(lambda a: a[1])(other_agent_batches)


def add_all_agents_gae(policy, sample_batch, other_agent_batches=None, episode=None):
    sample_batch = add_opponent_information_and_critical_vf(policy, sample_batch, other_agent_batches, episode=episode)

    global value_normalizer

    if value_normalizer.updated:
        sample_batch[SampleBatch.VF_PREDS] = value_normalizer.denormalize(sample_batch[SampleBatch.VF_PREDS])

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
    sample_batch = add_all_agents_gae(policy, sample_batch, other_agent_batches, episode)

    last_r = _get_last_r(policy, sample_batch)
    gamma = policy.config["gamma"]

    sample_batch = _add_returns(sample_batch=sample_batch, last_r=last_r, gamma=gamma)
    sample_batch = _add_deltas(sample_batch=sample_batch, last_r=last_r, gamma=gamma)

    return sample_batch


def hatrpo_post_process(policy, sample_batch, other_agent_batches=None, epsisode=None):
    sample_batch = trpo_post_process(policy, sample_batch, other_agent_batches, epsisode)

    n_agents = get_agent_num(policy)

    for i in range(n_agents - 1):
        cur_training = 0

        if other_agent_batches:
            name = exist_in_opponent(opponent_index=i, opponent_batches=other_agent_batches)
            if name:
                _p, _b = other_agent_batches[name]
                cur_training = _b.is_training

        sample_batch[get_global_name(TRAINING, i)] = np.array([
                                                                  int(cur_training)
                                                              ] * len(sample_batch))

    return sample_batch


def contain_global_obs(train_batch):
    return any(key.startswith(GLOBAL_PREFIX) for key in train_batch)


def get_action_from_batch(train_batch, model, dist_class):
    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    return curr_action_dist


def add_other_agent_mul_info(sample_batch, other_agent_info, agent_num):
    global GLOBAL_NEED_COLLECT

    for agent_i in range(agent_num - 1):
        for key in GLOBAL_NEED_COLLECT:
            if key not in sample_batch: continue

            value_added = False

            if other_agent_info:
                name = exist_in_opponent(opponent_index=agent_i, opponent_batches=other_agent_info)
                if name:
                    _p, _b = other_agent_info[name]
                    sample_batch[get_global_name(key, agent_i)] = _b[key]
                    value_added = True

            if not value_added:
                sample_batch[get_global_name(key, agent_i)] = np.zeros_like(
                    sample_batch[key],
                    dtype=sample_batch[key].dtype
                )

    if SampleBatch.ACTIONS in GLOBAL_NEED_COLLECT:
        sample_batch[get_global_name(SampleBatch.ACTIONS)] = np.stack([
            sample_batch[get_global_name(SampleBatch.ACTIONS, a_i)]
            for a_i in range(agent_num - 1)
        ], axis=1)

    return sample_batch


def get_vf_pred(policy, algorithm, sample_batch, opp_action_in_cc):
    if algorithm in ["coma"]:
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
            convert_to_torch_tensor(
                sample_batch[STATE], policy.device),
            convert_to_torch_tensor(
                sample_batch[get_global_name(SampleBatch.ACTIONS)], policy.device) if opp_action_in_cc else None,
        ) \
            .cpu().detach().numpy()
        sample_batch[SampleBatch.VF_PREDS] = np.take(
            sample_batch[SampleBatch.VF_PREDS],
            np.expand_dims(sample_batch[SampleBatch.ACTIONS], axis=1)
        ).squeeze(axis=1)
    else:
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
            convert_to_torch_tensor(
                sample_batch[STATE], policy.device),
            convert_to_torch_tensor(
                sample_batch[get_global_name(SampleBatch.ACTIONS)], policy.device) if opp_action_in_cc else None,
        ) \
            .cpu().detach().numpy()

    return sample_batch


def get_real_state_by_one_sample():
    pass


def collect_opponent_array(other_agent_batches: dict, opponent_agents_num, sample_batch):
    assert other_agent_batches is not None
    opponent_batch_list = list(other_agent_batches.values())
    raw_opponent_batch = [opponent_batch_list[i][1] for i in range(opponent_agents_num)]

    opponent_batch = []

    for i, one_opponent_batch in enumerate(raw_opponent_batch):
        if len(one_opponent_batch) != len(sample_batch):
            if len(one_opponent_batch) > len(sample_batch):
                one_opponent_batch = one_opponent_batch.slice(0, len(sample_batch))
            else:  # len(one_opponent_batch) < len(sample_batch):
                length_dif = len(sample_batch) - len(one_opponent_batch)
                one_opponent_batch = one_opponent_batch.concat(
                    one_opponent_batch.slice(len(one_opponent_batch) - length_dif, len(one_opponent_batch)))

        opponent_batch.append(one_opponent_batch)

    return opponent_batch


def state_name(i): return f'state_in_{i}'


def global_state_name(i, ai): return f'state_in_{i}_of_agent_{ai}'


def exist_in_opponent(opponent_index, opponent_batches: dict):
    possible_1 = f'agent_{opponent_index}'
    possible_2 = f'adversary_{opponent_index}'

    if possible_1 in opponent_batches:
        return possible_1
    elif possible_2 in opponent_batches:
        return possible_2
    else:
        return False


def add_state_in_for_opponent(sample_batch, other_agent_batches, agent_num):
    state_in_num = 0

    while state_name(state_in_num) in sample_batch:
        state_in_num += 1

    # get how many state in layer
    # agent_indices = [0, 1, 2]

    for a_i in range(agent_num - 1):
        # name = agent(a_i)

        for s_i in range(state_in_num):
            opponent_state_exist = False
            if other_agent_batches:
                name = exist_in_opponent(opponent_index=a_i, opponent_batches=other_agent_batches)
                if name and state_name(s_i) in other_agent_batches[name]:
                    _policy, _batch = other_agent_batches[name]
                    sample_batch[global_state_name(s_i, a_i)] = _batch[state_name(s_i)]
                    opponent_state_exist = True

            if not opponent_state_exist:
                sample_batch[global_state_name(s_i, a_i)] = np.zeros_like(
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
