"""
Implement HAPPO algorithm based on Rlib original PPO.
__author__: minquan
__data__: March-29-2022
"""

import logging
from typing import Dict, List, Type, Union, Tuple
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.agents.ppo.ppo_torch_policy import ppo_surrogate_loss as torch_loss
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    explained_variance, sequence_mask
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
import numpy as np
from ray.rllib.agents.ppo.ppo_tf_policy import (
    ppo_surrogate_loss as tf_loss,
)
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing, compute_gae_for_sample_batch
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf, try_import_torch, get_variable
from ray.rllib.utils.torch_ops import convert_to_torch_tensor as _d2t
from torch import nn
from MaMujoco.util.valuenorm import ValueNorm
from ray.rllib.utils.typing import TrainerConfigDict, TensorType, \
    LocalOptimizer
from MaMujoco.util.trpo_utilities import (
    conjugate_gradient,
    hessian_vector_product
)

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


value_normalizer = ValueNorm(1)

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

"""
def postprocess_fn(policy, sample_batch, other_agent_batches, episode):
    agents = ["agent_1", "agent_2", "agent_3"]  # simple example of 3 agents
    global_obs_batch = np.stack(
        [other_agent_batches[agent_id][1]["obs"] for agent_id in agents],
        axis=1)
    # add the global obs and global critic value
    sample_batch["global_obs"] = global_obs_batch
    sample_batch["central_vf"] = self.sess.run(
        self.critic_network, feed_dict={"obs": global_obs_batch})
    return sample_batch

"""
from ray.rllib.examples.centralized_critic import CentralizedValueMixin


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.

GLOBAL_NEED_COLLECT = [SampleBatch.ACTION_LOGP, SampleBatch.ACTIONS]
GLOBAL_PREFIX = 'GLOBAL_'
GLOBAL_MODEL_LOGITS = f'{GLOBAL_PREFIX}_model_logits'
STATE = 'state'
convert_to_torch_tensor = _d2t


def add_other_agent_info(agents_batch: dict, key: str):
    # get other-agents information by specific key

    _POLICY_INDEX, _BATCH_INDEX = 0, 1

    agent_ids = sorted([agent_id for agent_id in agents_batch])

    return np.stack([
        agents_batch[agent_id][_BATCH_INDEX][key] for agent_id in agent_ids],
        axis=1
    )


def get_global_name(key):
    # converts a key to global format

    return f'{GLOBAL_PREFIX}{key}'


def collect_other_agents_model_output(agents_batch):

    # other_agents_logits = []

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


def add_another_agent_and_gae(policy, sample_batch, other_agent_batches=None, episode=None):
    # train_batch = centralized_critic_postprocessing(policy, sample_batch, other_agent_batches, episode)
    global value_normalizer

    if value_normalizer.updated:
        sample_batch[SampleBatch.VF_PREDS] = value_normalizer.denormalize(sample_batch[SampleBatch.VF_PREDS])

    train_batch = compute_gae_for_sample_batch(policy, sample_batch, other_agent_batches, episode)

    state_dim = policy.config["model"]["custom_model_config"]["state_dim"]

    sample_batch[STATE] = sample_batch[SampleBatch.OBS][:, -state_dim:]

    if other_agent_batches:
        for key in GLOBAL_NEED_COLLECT:
            train_batch[get_global_name(key)] = add_other_agent_info(agents_batch=other_agent_batches, key=key)

        train_batch[GLOBAL_MODEL_LOGITS] = collect_other_agents_model_output(other_agent_batches)

    return train_batch


def contain_global_obs(train_batch):
    return any(key.startswith(GLOBAL_PREFIX) for key in train_batch)


def ppo_surrogate_for_one_agent(importance_sampling, advantage, epsilon):
    surrogate_loss = torch.min(
        advantage * importance_sampling,
        advantage * torch.clamp(
            importance_sampling, 1 - epsilon,
            1 + epsilon
        )
    )

    return surrogate_loss


def trpo_surrogate_for_one_agent(advantages, log_probs):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    surr_loss = -(log_probs * advantages)

    return surr_loss


def get_action_from_batch(train_batch, model, dist_class):
    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    return curr_action_dist


PPO, TRPO = 'PPO', 'TRPO'


def surrogate_loss_for_ppo_and_trpo(run: str):
    assert run.upper() in (PPO, TRPO)

    def wrap(policy, model, dist_class, train_batch):
        ppo_surrogate_loss(policy, model, dist_class, train_batch, run=run.upper())

    return wrap


def get_loss(run, advantages, importance_sampling, log_probs, eps):
    if run == PPO:
        sub_loss = ppo_surrogate_for_one_agent(importance_sampling=importance_sampling,
                                               advantage=advantages,
                                               epsilon=eps)
    elif run == TRPO:
        sub_loss = trpo_surrogate_for_one_agent(advantages=advantages, log_probs=log_probs)
    else:
        raise TypeError('unsupported algorithm type')

    return sub_loss


def ppo_surrogate_loss(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch, run='PPO') -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for Proximal Policy Objective.
    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.
    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """

    CentralizedValueMixin.__init__(policy)

    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # print(f'agent-{train_batch[SampleBatch.AGENT_INDEX][0]} with advantage: {torch.mean(train_batch[Postprocessing.ADVANTAGES])}')

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major())
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean

    vf_saved = model.value_function

    if contain_global_obs(train_batch):
        model.value_function = lambda: policy.model.central_value_function(
            train_batch[STATE], train_batch[get_global_name(SampleBatch.ACTIONS)])

        policy._central_value_out = model.value_function()  # change value function to calculate all agents information

        sub_losses = []

        m_advantage = train_batch[Postprocessing.ADVANTAGES]

        agents_num = train_batch[GLOBAL_MODEL_LOGITS].shape[1] + 1
        # all_agents = [SELF] + train_batch[get_global_name(SampleBatch.OBS)]

        random_indices = np.random.permutation(range(agents_num))
        # print(f'there are {agents_num} agents, training as {random_indices}')
        # in order to get each agent's information, if random_indices is len(agents_num) - 1, we set
        # this as our current_agent, and get the information from generally train batch.
        # otherwise, we get the agent information from "GLOBAL_LOGITS", "GLOBAL_ACTIONS", etc

        def is_current_agent(i): return i == agents_num - 1

        torch.autograd.set_detect_anomaly(True)

        for agent_id in random_indices:
            if is_current_agent(agent_id):
                logits, state = model(train_batch)
                current_action_dist = dist_class(logits, model)
                old_action_log_dist = train_batch[SampleBatch.ACTION_LOGP]
                actions = train_batch[SampleBatch.ACTIONS]
                log_prob = current_action_dist.logp(actions)
            else:
                current_action_logits = train_batch[GLOBAL_MODEL_LOGITS][:, agent_id, :].detach()
                current_action_dist = dist_class(current_action_logits, None)
                # current_action_dist = train_batch[GLOBAL_MODEL_LOGITS][:, agent_id, :]
                old_action_log_dist = train_batch[get_global_name(SampleBatch.ACTION_LOGP)][:, agent_id].detach()
                actions = train_batch[get_global_name(SampleBatch.ACTIONS)][:, agent_id, :].detach()
                log_prob = old_action_log_dist

            importance_sampling = torch.exp(current_action_dist.logp(actions) - old_action_log_dist)

            m_advantage = importance_sampling * m_advantage

            sub_loss = get_loss(run=run, advantages=m_advantage, importance_sampling=importance_sampling, log_probs=log_prob,
                                eps=policy.config['clip_param'])

            sub_losses.append(sub_loss)

        surrogate_loss = torch.mean(torch.stack(sub_losses, axis=1), axis=1)
    else:
        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) -
            train_batch[SampleBatch.ACTION_LOGP])

        surrogate_loss = get_loss(run, advantages=train_batch[Postprocessing.ADVANTAGES], importance_sampling=logp_ratio,
                                  eps=policy.config['clip_param'], log_probs=curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]))

    mean_policy_loss = reduce_mean_valid(-surrogate_loss)

    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)
    action_kl = prev_action_dist.kl(curr_action_dist)
    mean_kl_loss = reduce_mean_valid(action_kl)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    # Compute a value function loss.
    if policy.model.model_config['custom_model_config']['normal_value']:
        value_normalizer.update(train_batch[Postprocessing.VALUE_TARGETS])
        train_batch[Postprocessing.VALUE_TARGETS] = value_normalizer.normalize(train_batch[Postprocessing.VALUE_TARGETS])

    if policy.config["use_critic"]:
        prev_value_fn_out = train_batch[SampleBatch.VF_PREDS] #
        value_fn_out = model.value_function()  # same as values
        vf_loss1 = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_clipped = prev_value_fn_out + torch.clamp(
            value_fn_out - prev_value_fn_out, -policy.config["vf_clip_param"],
            policy.config["vf_clip_param"])
        vf_loss2 = torch.pow(
            vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_loss = torch.max(vf_loss1, vf_loss2)
        mean_vf_loss = reduce_mean_valid(vf_loss)
    # Ignore the value function.
    else:
        vf_loss = mean_vf_loss = 0.0

    model.value_function = vf_saved
    # recovery the value function.

    total_loss = reduce_mean_valid(-surrogate_loss +
                                   policy.kl_coeff * action_kl +
                                   policy.config["vf_loss_coeff"] * vf_loss -
                                   policy.entropy_coeff * curr_entropy)

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], model.value_function())
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss


def make_happo_optimizers(policy: Policy,
                          config: TrainerConfigDict) -> Tuple[LocalOptimizer]:
    """Create separate optimizers for actor & critic losses."""

    # Set epsilons to match tf.keras.optimizers.Adam's epsilon default.
    policy._actor_optimizer = torch.optim.Adam(
        params=policy.model.policy_variables(),
        lr=config["actor_lr"],
        eps=1e-7)

    policy._critic_optimizer = torch.optim.Adam(
        params=policy.model.critic_variables(), lr=config["critic_lr"], eps=1e-5)

    # Return them in the same order as the respective loss terms are returned.
    return policy._actor_optimizer, policy._critic_optimizer


def grad_extra_for_trpo(policy, optimizer, loss):
    info = {}

    for param_group in optimizer.param_groups:
        # Make sure we only pass params with grad != None into torch
        # clip_grad_norm_. Would fail otherwise.
        params = list(
            filter(lambda p: p.grad is not None, param_group["params"]))

        if params:
            grad_norm = nn.utils.clip_grad_norm(params, policy.config['grad_clip'])
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.cpu().numpy()
            info['grad_norm'] = grad_norm

    return info


def _compute_descent_step(policy: Policy, pol_grad, obs):
    """Approximately compute the Natural gradient using samples.

    This is based on the Fisher Matrix formulation as the hessian of the average
    entropy. For more information, see:
    https://en.wikipedia.org/wiki/Fisher_information#Matrix_form

    Args:
        pol_grad (Tensor): The vector to compute the Fisher vector product with.
        obs (Tensor): The observations to evaluate the policy in.
    """
    params = list(policy.module.actor.parameters())

    fvp_samples = 10
    cg_damping = 1e-3
    cg_iters = 10

    with torch.no_grad():
        ent_acts, _ = policy.module.actor.sample(obs, fvp_samples)

    def entropy():
        # return policy.compute_log_likelihoods()
        return policy.module.actor.log_prob(obs, ent_acts).neg().mean()

    def fvp(vec):
        return hessian_vector_product(entropy(), params, vec)

    descent_direction, elapsed_iters, residual = conjugate_gradient(
        lambda x: fvp(x) + cg_damping * x,
        pol_grad,
        cg_iters=cg_iters,
    )

    fisher_norm = pol_grad.dot(descent_direction)
    delta = 1e-2
    scale = 0 if fisher_norm < 0 else torch.sqrt(2 * delta / (fisher_norm + 1e-8))

    descent_direction = descent_direction * scale
    return descent_direction
