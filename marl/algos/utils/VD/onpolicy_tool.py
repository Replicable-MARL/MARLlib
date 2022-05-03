from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.numpy import convert_to_numpy
from typing import Dict, List, Type, Union
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, \
    Postprocessing, compute_advantages
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    explained_variance, sequence_mask
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG, A2CTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, KLCoeffMixin, ValueNetworkMixin
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from copy import deepcopy
import numpy as np

torch, nn = try_import_torch()


class MixingValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        self.mixing_vf = "mixing"


def get_dim(a):
    dim = 1
    for i in a:
        dim *= i
    return dim


# get opponent value vf
def value_mix_centralized_critic_postprocessing(policy,
                                                sample_batch,
                                                other_agent_batches=None,
                                                episode=None):
    custom_config = policy.config["model"]["custom_model_config"]
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


# value decomposition based actor critic loss
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


VDA2CTorchPolicy = A3CTorchPolicy.with_updates(
    name="VDA2CTorchPolicy",
    get_default_config=lambda: A2C_CONFIG,
    postprocess_fn=value_mix_centralized_critic_postprocessing,
    loss_fn=value_mix_actor_critic_loss,
    mixins=[ValueNetworkMixin, MixingValueMixin],
)


def get_policy_class_vda2c(config_):
    if config_["framework"] == "torch":
        return VDA2CTorchPolicy


VDA2CTrainer = A2CTrainer.with_updates(
    name="VDA2CTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_vda2c,
)


# value decomposition based ppo loss
def value_mix_ppo_surrogate_loss(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
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
    MixingValueMixin.__init__(policy)

    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

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

    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS],
                                  model)

    logp_ratio = torch.exp(
        curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) -
        train_batch[SampleBatch.ACTION_LOGP])
    action_kl = prev_action_dist.kl(curr_action_dist)
    mean_kl_loss = reduce_mean_valid(action_kl)

    curr_entropy = curr_action_dist.entropy()
    mean_entropy = reduce_mean_valid(curr_entropy)

    surrogate_loss = torch.min(
        train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
        train_batch[Postprocessing.ADVANTAGES] * torch.clamp(
            logp_ratio, 1 - policy.config["clip_param"],
                        1 + policy.config["clip_param"]))
    mean_policy_loss = reduce_mean_valid(-surrogate_loss)

    # Compute a value function loss.
    if policy.config["use_critic"]:
        prev_value_fn_out = train_batch[SampleBatch.VF_PREDS]
        value_fn_out = model.value_function()

        # add mixing_function
        opponent_vf_preds = convert_to_torch_tensor(train_batch["opponent_vf_preds"])
        vf_pred = value_fn_out.unsqueeze(1)
        all_vf_pred = torch.cat((vf_pred, opponent_vf_preds), 1)
        state = convert_to_torch_tensor(train_batch["state"])
        value_tot = model.mixing_value(all_vf_pred, state)

        vf_loss1 = torch.pow(
            value_tot - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_clipped = prev_value_fn_out + torch.clamp(
            value_tot - prev_value_fn_out, -policy.config["vf_clip_param"],
            policy.config["vf_clip_param"])
        vf_loss2 = torch.pow(
            vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_loss = torch.max(vf_loss1, vf_loss2)
        mean_vf_loss = reduce_mean_valid(vf_loss)
    # Ignore the value function.
    else:
        vf_loss = mean_vf_loss = 0.0

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


VDPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="VDPPOTorchPolicy",
    get_default_config=lambda: PPO_CONFIG,
    postprocess_fn=value_mix_centralized_critic_postprocessing,
    loss_fn=value_mix_ppo_surrogate_loss,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin, MixingValueMixin
    ])


def get_policy_class_vdppo(config_):
    if config_["framework"] == "torch":
        return VDPPOTorchPolicy


VDPPOTrainer = PPOTrainer.with_updates(
    name="VDPPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_vdppo,
)
