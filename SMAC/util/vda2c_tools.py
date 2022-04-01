from ray.rllib.evaluation.postprocessing import Postprocessing, discount_cumsum
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, \
    Postprocessing
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.torch_ops import apply_grad_clipping, sequence_mask
from ray.rllib.utils.typing import TrainerConfigDict, TensorType, \
    PolicyID, LocalOptimizer
import numpy as np
import scipy.signal
from typing import Dict, Optional
from ray.rllib.utils.numpy import convert_to_numpy

from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.typing import AgentID
import numpy as np
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable
from ray.rllib.agents.ppo.ppo_torch_policy import ValueNetworkMixin

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

class MixingValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        self.mixing_vf = "mixing"

# additionally get value mixing v_tot here
def value_mix_centralized_critic_postprocessing(policy,
                                                sample_batch,
                                                other_agent_batches=None,
                                                episode=None):
    pytorch = policy.config["framework"] == "torch"
    self_obs_dim = policy.config["self_obs_dim"]
    state_dim = policy.config["state_dim"]
    n_agents = policy.config["agent_num"]
    opponent_agents_num = n_agents - 1

    if (pytorch and hasattr(policy, "mixing_vf")) or \
            (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None
        opponent_batch_list = list(other_agent_batches.values())

        # TODO sample batch size not equal across different batches.
        # here we only provide a solution to force the same length with sample batch
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

        sample_batch["self_obs"] = sample_batch['obs'][:, :self_obs_dim]
        sample_batch["state"] = sample_batch['obs'][:, self_obs_dim:self_obs_dim + state_dim]
        sample_batch["opponent_vf_preds"] = np.stack(
            [opponent_batch[i]["vf_preds"] for i in range(opponent_agents_num)],
            1)
        sample_batch["all_vf_preds"] = np.concatenate(
            (np.expand_dims(sample_batch["vf_preds"], axis=1), sample_batch["opponent_vf_preds"]), axis=1)

        sample_batch["vf_tot"] = convert_to_numpy(policy.model.mixing_value(
            convert_to_torch_tensor(sample_batch["all_vf_preds"]).cuda(),
            convert_to_torch_tensor(sample_batch["state"]).cuda()))

    else:
        # Policy hasn't been initialized yet, use zeros.
        o = sample_batch[SampleBatch.CUR_OBS]
        sample_batch["self_obs"] = np.zeros((o.shape[0], self_obs_dim),
                                            dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
        sample_batch["state"] = np.zeros((o.shape[0], state_dim),
                                         dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
        sample_batch["vf_preds"] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)

        sample_batch["opponent_vf_preds"] = np.zeros(
            (sample_batch["vf_preds"].shape[0], opponent_agents_num),
            dtype=sample_batch["obs"].dtype)

        sample_batch["all_vf_preds"] = np.concatenate(
            (np.expand_dims(sample_batch["vf_preds"], axis=1), sample_batch["opponent_vf_preds"]), axis=1)

        sample_batch["vf_tot"] = convert_to_numpy(policy.model.mixing_value(
            convert_to_torch_tensor(sample_batch["all_vf_preds"]).cuda(),
            convert_to_torch_tensor(sample_batch["state"]).cuda()))

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
    """
    Given a rollout, compute its value targets and the advantages.

    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory.
        last_r (float): Value estimation for last observation.
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE.
        use_gae (bool): Using Generalized Advantage Estimation.
        use_critic (bool): Whether to use critic (value estimates). Setting
            this to False will use 0 as baseline.

    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """

    assert SampleBatch.VF_PREDS in rollout or not use_critic, \
        "use_critic=True but values not found"
    assert use_critic or not use_gae, \
        "Can't use gae without using a value function"

    if use_gae:
        # if np.array([last_r]).size == 0 or rollout["vf_tot"].size == 0:
        #     print(1)
        vpred_t = np.concatenate(
            [rollout["vf_tot"],
             np.array([last_r])])
        delta_t = (
                rollout[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        rollout[Postprocessing.ADVANTAGES] = discount_cumsum(
            delta_t, gamma * lambda_)
        rollout[Postprocessing.VALUE_TARGETS] = (
                rollout[Postprocessing.ADVANTAGES] +
                rollout["vf_tot"]).astype(np.float32)
    else:
        rewards_plus_v = np.concatenate(
            [rollout[SampleBatch.REWARDS],
             np.array([last_r])])
        discounted_returns = discount_cumsum(rewards_plus_v,
                                             gamma)[:-1].astype(np.float32)

        if use_critic:
            rollout[Postprocessing.
                ADVANTAGES] = discounted_returns - rollout["vf_tot"]
            rollout[Postprocessing.VALUE_TARGETS] = discounted_returns
        else:
            rollout[Postprocessing.ADVANTAGES] = discounted_returns
            rollout[Postprocessing.VALUE_TARGETS] = np.zeros_like(
                rollout[Postprocessing.ADVANTAGES])

    rollout[Postprocessing.ADVANTAGES] = rollout[
        Postprocessing.ADVANTAGES].astype(np.float32)

    return rollout


def value_mix_actor_critic_loss(policy: Policy, model: ModelV2,
                                dist_class: ActionDistribution,
                                train_batch: SampleBatch) -> TensorType:
    MixingValueMixin.__init__(policy)

    logits, _ = model.from_batch(train_batch)
    values = model.value_function()

    # add mixing_function
    opponent_vf_preds = convert_to_torch_tensor(train_batch["opponent_vf_preds"])
    vf_pred = values.unsqueeze(1)
    all_vf_pred = torch.cat((vf_pred, opponent_vf_preds), 1)
    state = convert_to_torch_tensor(train_batch["state"])
    value_tot = model.mixing_value(all_vf_pred, state)

    if policy.is_recurrent():
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask_orig = sequence_mask(train_batch[SampleBatch.SEQ_LENS],
                                  max_seq_len)
        valid_mask = torch.reshape(mask_orig, [-1])
    else:
        valid_mask = torch.ones_like(value_tot, dtype=torch.bool)

    dist = dist_class(logits, model)
    log_probs = dist.logp(train_batch[SampleBatch.ACTIONS]).reshape(-1)
    pi_err = -torch.sum(
        torch.masked_select(log_probs * train_batch[Postprocessing.ADVANTAGES],
                            valid_mask))

    # Compute a value function loss.
    if policy.config["use_critic"]:
        value_err = 0.5 * torch.sum(
            torch.pow(
                torch.masked_select(
                    value_tot.reshape(-1) -
                    train_batch[Postprocessing.VALUE_TARGETS], valid_mask),
                2.0))
    # Ignore the value function.
    else:
        value_err = 0.0

    entropy = torch.sum(torch.masked_select(dist.entropy(), valid_mask))

    total_loss = (pi_err + value_err * policy.config["vf_loss_coeff"] -
                  entropy * policy.config["entropy_coeff"])

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["entropy"] = entropy
    model.tower_stats["pi_err"] = pi_err
    model.tower_stats["value_err"] = value_err

    return total_loss
