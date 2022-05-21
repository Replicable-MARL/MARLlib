from ray.rllib.utils.torch_ops import convert_to_torch_tensor as _d2t
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np
from ray.rllib.evaluation.postprocessing import discount_cumsum, Postprocessing, compute_gae_for_sample_batch
from marl.algos.utils.valuenorm import ValueNorm


GLOBAL_NEED_COLLECT = [SampleBatch.ACTION_LOGP, SampleBatch.ACTIONS]
GLOBAL_PREFIX = 'GLOBAL_'
GLOBAL_MODEL_LOGITS = f'{GLOBAL_PREFIX}_model_logits'
GLOBAL_MODEL = f'{GLOBAL_PREFIX}_model'
GLOBAL_TRAIN_BATCH = f'{GLOBAL_PREFIX}_train_batch'
# STATE = 'state'


value_normalizer = ValueNorm(1)


def add_other_agent_info(agents_batch: dict, key: str):
    # get other-agents information by specific key

    _POLICY_INDEX, _BATCH_INDEX = 0, 1

    return np.stack([
        agents_batch[agent_id][_BATCH_INDEX][key] for agent_id in agents_batch],
        axis=1
    )


def get_global_name(key):
    # converts a key to global format

    return f'{GLOBAL_PREFIX}{key}'


def collect_other_agents_model_output(agents_batch):
    agent_ids = sorted([agent_id for agent_id in agents_batch])

    other_agents_logits = np.stack([
        agents_batch[_id][1][SampleBatch.ACTION_DIST_INPUTS] for _id in agent_ids
    ], axis=1)

    # for agent_id, (policy, obs) in agents_batch.items():
        # agent_model = policy.model
        # assert isinstance(obs, SampleBatch)
        # agent_logits, state = agent_model(_d2t(obs))
        # dis_class = TorchDistributionWrapper
        # curr_action_dist = dis_class(agent_logits, agent_model)
        # action_log_dist = curr_action_dist.logp(obs[SampleBatch.ACTIONS])

        # other_agents_logits.append(obs[SampleBatch.ACTION_DIST_INPUTS])

    # other_agents_logits = np.stack(other_agents_logits, axis=1)

    return other_agents_logits


def _extract_from_all_others(take_fn):
    def inner(other_agent_batches):
        agent_ids = sorted([agent_id for agent_id in other_agent_batches])

        return [
            take_fn(other_agent_batches[_id]) for _id in agent_ids
        ]

    return inner


def extract_all_other_agents_model(other_agent_batches):
    return _extract_from_all_others(lambda a: a[0].model)(other_agent_batches)


def extract_other_agents_train_batch(other_agent_batches):
    return _extract_from_all_others(lambda a: a[1])(other_agent_batches)


def add_all_agents_gae(policy, sample_batch, other_agent_batches=None, episode=None):
    # train_batch = centralized_critic_postprocessing(policy, sample_batch, other_agent_batches, episode)
    global value_normalizer

    if value_normalizer.updated:
        sample_batch[SampleBatch.VF_PREDS] = value_normalizer.denormalize(sample_batch[SampleBatch.VF_PREDS])

    train_batch = compute_gae_for_sample_batch(policy, sample_batch, other_agent_batches, episode)

    # state_dim = policy.config["model"]["custom_model_config"]["state_dim"]
    # sample_batch[STATE] = sample_batch[SampleBatch.OBS][:, -state_dim:]
    # sample_batch[STATE] = sample_batch[SampleBatch.OBS]

    if other_agent_batches:
        for key in GLOBAL_NEED_COLLECT:
            train_batch[get_global_name(key)] = add_other_agent_info(agents_batch=other_agent_batches, key=key)

        train_batch[GLOBAL_MODEL_LOGITS] = collect_other_agents_model_output(other_agent_batches)

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


def trpo_post_process(policy, sample_batch, other_agent_batches=None, epsisode=None):
    sample_batch = add_all_agents_gae(policy, sample_batch, other_agent_batches, epsisode)

    last_r = _get_last_r(policy, sample_batch)
    gamma = policy.config["gamma"]

    sample_batch = _add_returns(sample_batch=sample_batch, last_r=last_r, gamma=gamma)
    sample_batch = _add_deltas(sample_batch=sample_batch, last_r=last_r, gamma=gamma)

    return sample_batch


def hatrpo_post_process(policy, sample_batch, other_agent_batches=None, epsisode=None):

    sample_batch = trpo_post_process(policy, sample_batch, other_agent_batches, epsisode)

    if other_agent_batches:
        models = extract_all_other_agents_model(other_agent_batches=other_agent_batches)
        sample_batch[GLOBAL_MODEL] = models

        train_batches = extract_other_agents_train_batch(other_agent_batches=other_agent_batches)
        sample_batch[GLOBAL_TRAIN_BATCH] = train_batches

    return sample_batch


def contain_global_obs(train_batch):
    return any(key.startswith(GLOBAL_PREFIX) for key in train_batch)


def get_action_from_batch(train_batch, model, dist_class):
    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    return curr_action_dist

