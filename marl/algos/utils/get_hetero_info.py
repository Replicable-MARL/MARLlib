from ray.rllib.utils.torch_ops import convert_to_torch_tensor as _d2t
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np
from ray.rllib.evaluation.postprocessing import discount_cumsum, Postprocessing, compute_gae_for_sample_batch
from marl.algos.utils.valuenorm import ValueNorm
from copy import deepcopy
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from icecream import ic
from marl.algos.utils.postprocessing import get_dim, convert_to_torch_tensor, centralized_critic_postprocessing
from ray.rllib.evaluation.postprocessing import compute_advantages
import re


GLOBAL_NEED_COLLECT = [SampleBatch.ACTION_LOGP, SampleBatch.ACTIONS,
                       SampleBatch.ACTION_DIST_INPUTS, SampleBatch.SEQ_LENS]
GLOBAL_PREFIX = 'opponent_'
GLOBAL_MODEL_LOGITS = f'{GLOBAL_PREFIX}model_logits'
GLOBAL_MODEL = f'{GLOBAL_PREFIX}model'
GLOBAL_IS_TRAINING = f'{GLOBAL_PREFIX}is_trainable'
GLOBAL_TRAIN_BATCH = f'{GLOBAL_PREFIX}train_batch'
STATE = 'state'
# STATE = 'state'


value_normalizer = ValueNorm(1)


def get_global_name(key):
    # converts a key to global format

    return f'{GLOBAL_PREFIX}{key}'


# def collect_other_agents_model_output(agents_batch):
#     agent_ids = sorted([agent_id for agent_id in agents_batch])

    # other_agents_logits = np.stack([
    #     agents_batch[_id][1][SampleBatch.ACTION_DIST_INPUTS] for _id in agent_ids
    # ], axis=1)

    # for agent_id, (policy, obs) in agents_batch.items():
        # agent_model = policy.model
        # assert isinstance(obs, SampleBatch)
        # agent_logits, state = agent_model(_d2t(obs))
        # dis_class = TorchDistributionWrapper
        # curr_action_dist = dis_class(agent_logits, agent_model)
        # action_log_dist = curr_action_dist.logp(obs[SampleBatch.ACTIONS])

        # other_agents_logits.append(obs[SampleBatch.ACTION_DIST_INPUTS])

    # other_agents_logits = np.stack(other_agents_logits, axis=1)

    # return other_agents_logits


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

    # state_dim = policy.config["model"]["custom_model_config"]["state_dim"]
    # sample_batch[STATE] = sample_batch[SampleBatch.OBS][:, -state_dim:]
    # sample_batch[STATE] = sample_batch[SampleBatch.OBS]

    # if other_agent_batches:
    #     for key in GLOBAL_NEED_COLLECT:
    #         train_batch[get_global_name(key)] = add_other_agent_info(agents_batch=other_agent_batches, key=key)

        # train_batch[GLOBAL_MODEL_LOGITS] = collect_other_agents_model_output(other_agent_batches)

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

    if other_agent_batches:

        for agent_id, (_policy, _samply_batch)  in other_agent_batches.items():
            _id = re.findall(r'\d+', agent_id)[0]
            _id = int(_id)

            sample_batch['opponent_model_{}'.format(_id)] = np.array([int(id(_policy.model))], dtype=np.int)
            sample_batch['opponent_training_{}'.format(_id)] = np.array([int(_samply_batch.is_training)], dtype=np.int)

        # train_batches = extract_other_agents_train_batch(other_agent_batches=other_agent_batches)

        # train_batches = pad_batch_to_sequences_of_same_size(train_batches, max_seq_len=policy.config['max_len'])
        # train_batches = [(t, max_seq_len=policy.max_seq_len, multi_agent=True) for t in train_batches]

        # sample_batch[GLOBAL_STATE] = np.array([get_one_batch_state(b) for b in train_batches])
        # ic(sample_batch[GLOBAL_STATE].shape)
        # sample_batch[GLOBAL_IS_TRAINING] = np.stack([ [int(b.is_training)] for b in train_batches], axis=1)


        # sample_batch[GLOBAL_TRAIN_BATCH] = [t.copy(shallow=True) for t in train_batches]
        # sample_batch[GLOBAL_OBS] = [batch[SampleBatch.OBS] for batch in train_batches]
        # sample_batch[GLOBAL_ACTION_LOGP] = [batch[SampleBatch.ACTION_LOGP] for batch in train_batches]
        # sample_batch[GLOBAL_OBS] = [batch[SampleBatch.OBS] for batch in train_batches]

    return sample_batch


def contain_global_obs(train_batch):
    return any(key.startswith(GLOBAL_PREFIX) for key in train_batch)


def get_action_from_batch(train_batch, model, dist_class):
    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    return curr_action_dist


def add_other_agent_mul_info(agent_batch, to_initialize=False, sample_batch=None):
    global GLOBAL_NEED_COLLECT

    for key in GLOBAL_NEED_COLLECT:
        if key not in sample_batch: continue

        sample_batch[get_global_name(key)] = add_other_agent_one_info(
            agent_batch, key=key,
            to_initialize=to_initialize,
            sample_batch=sample_batch
        )

    return sample_batch


def add_mul_opponents_agent_real_obs_info(opponent_batches: list, obs_dim, action_mask_dim, global_state_flag,
                                          sample_batch,
                                          initialize=False):
    if initialize:
        sample_batch[get_global_name(SampleBatch.OBS)] = np.stack([
            np.zeros_like(sample_batch[SampleBatch.OBS], dtype=sample_batch[STATE].dtype)for _ in range(len(opponent_batches))
        ], axis=1)
    else:
        if global_state_flag:
            sample_batch[get_global_name(SampleBatch.OBS)] = np.stack([
                sample_batch[SampleBatch.OBS] for _ in range(len(opponent_batches))
            ], axis=1)
        else:
            # action_mask_dim = action_mask_dim or 0   # set to zero if action is None
            # start = action_mask_dim

            sample_batch[get_global_name(SampleBatch.OBS)] = np.stack([
                batch[SampleBatch.OBS] for batch in opponent_batches], axis=1
            )

    return sample_batch


def add_other_agent_one_info(opponent_batch: list, key: str, to_initialize=False, sample_batch=None,
                             opponent_agents_num=None,
                             ):
    # get other-agents information by specific key

    opponent_agents_num = opponent_agents_num or len(opponent_batch)

    if to_initialize:
        return np.stack([
            np.zeros_like(sample_batch[key], dtype=sample_batch[key].dtype) for _ in range(opponent_agents_num)
        ], axis=1)
    else:
        return np.stack([
            opponent_batch[i][key] for i in range(opponent_agents_num)],
            axis=1
        )


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
            np.expand_dims(sample_batch[SampleBatch.ACTIONS],axis=1)
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
    # agent_keys = sorted(other_agent_batches.keys())
    # raw_opponent_batch = [other_agent_batches[a_id][1] for a_id in agent_keys]
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


def add_state_in_for_opponent(sample_batch, other_agent_batches):

    def _name(i):return f'state_in_{i}'

    i = 0
    while _name(i) in sample_batch:
        sample_batch[get_global_name(_name(i))] = []
        for batch in other_agent_batches:
            if batch and _name(i) in batch:
                sample_batch[get_global_name(_name(i))].append(batch[_name(i)])
            else:
                sample_batch[get_global_name(_name(i))].append(np.zeros_like(sample_batch[_name(i)]))

        sample_batch[get_global_name(_name(i))] = np.stack(
            sample_batch[get_global_name(_name(i))], axis=1
        )

        i += 1

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

    n_agents = custom_config["num_agents"]
    opponent_agents_num = n_agents - 1

    opponent_info_exists = (pytorch and hasattr(policy, "compute_central_vf")) or (not pytorch and policy.loss_initialized())

    opponent_batch = [None] * opponent_agents_num

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
            # sample_batch["state"] = sample_batch['obs'][:, action_mask_dim:action_mask_dim + obs_dim]
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


    need_initialize_opponent_info = not opponent_info_exists

    sample_batch = add_other_agent_mul_info(
        agent_batch=opponent_batch,
        to_initialize=need_initialize_opponent_info,
        sample_batch=sample_batch,
    )

    sample_batch = add_mul_opponents_agent_real_obs_info(
        opponent_batches=opponent_batch,
        obs_dim=obs_dim,
        action_mask_dim=action_mask_dim,
        global_state_flag=global_state_flag,
        initialize=need_initialize_opponent_info,
        sample_batch=sample_batch,
    )

    # sample_batch[SampleBatch.OBS] = sample_batch[STATE]

    sample_batch = add_state_in_for_opponent(
        sample_batch=sample_batch,
        other_agent_batches=opponent_batch,
    )

    if opponent_info_exists:
        sample_batch = get_vf_pred(policy, algorithm, sample_batch, opp_action_in_cc)
    else:
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    # ic(sample_batch[SampleBatch.OBS].shape)
    # ic(sample_batch[get_global_name(SampleBatch.OBS)].shape)

    # ic(sample_batch[get_global_name('state_in_0')].shape)
    # ic(sample_batch['obs'].shape)
    # ic(sample_batch[get_global_name('obs')].shape)
    # ic(sample_batch[get_global_name('actions')].shape)

    return sample_batch
