from ray.rllib.agents.ppo.ppo_torch_policy import ValueNetworkMixin
from ray.rllib.utils.torch_ops import apply_grad_clipping, sequence_mask
import logging
import gym
from typing import Dict, Tuple

import ray
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    concat_multi_gpu_td_errors, huber_loss, l2_loss
from ray.rllib.utils.typing import TrainerConfigDict, TensorType, \
    LocalOptimizer, GradInfoDict
from Hanabi.util.mappo_tools import centralized_critic_postprocessing, CentralizedValueMixin
from ray.rllib.agents.a3c.a3c_tf_policy import actor_critic_loss as tf_loss
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from gym.spaces.box import Box
from ray.rllib.evaluation.postprocessing import adjust_nstep
from ray.rllib.utils.numpy import convert_to_numpy
import numpy as np
from ray.rllib.utils.framework import try_import_tf, try_import_torch

torch, nn = try_import_torch()


def loss_with_central_critic_coma(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = coma_loss

    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(
        train_batch["obs"], train_batch["opponent_obs"],
        train_batch["opponent_action"])

    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)

    model.value_function = vf_saved

    return loss


def central_vf_stats_coma(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy._central_value_out)
    }


def coma_loss(policy: Policy, model: ModelV2,
              dist_class: ActionDistribution,
              train_batch: SampleBatch) -> TensorType:
    logits, _ = model.from_batch(train_batch)
    values = model.value_function()

    if policy.is_recurrent():
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask_orig = sequence_mask(train_batch[SampleBatch.SEQ_LENS],
                                  max_seq_len)
        valid_mask = torch.reshape(mask_orig, [-1])
    else:
        valid_mask = torch.ones_like(values, dtype=torch.bool)

    dist = dist_class(logits, model)
    log_probs = dist.logp(train_batch[SampleBatch.ACTIONS]).reshape(-1)

    # here the coma loss & calculate the mean values as baseline:
    select_action_Q_value = values.gather(1, train_batch[SampleBatch.ACTIONS].unsqueeze(1)).squeeze()
    advantages = (select_action_Q_value - torch.mean(values, dim=1)).detach()
    coma_pi_err = -torch.sum(torch.masked_select(log_probs * advantages, valid_mask))

    # Compute coma critic loss.
    if policy.config["use_critic"]:
        value_err = 0.5 * torch.sum(
            torch.pow(
                torch.masked_select(
                    select_action_Q_value.reshape(-1) -
                    train_batch[Postprocessing.VALUE_TARGETS], valid_mask),
                2.0))
    # Ignore the value function.
    else:
        value_err = 0.0

    entropy = torch.sum(torch.masked_select(dist.entropy(), valid_mask))

    total_loss = (coma_pi_err + value_err * policy.config["vf_loss_coeff"] -
                  entropy * policy.config["entropy_coeff"])

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["entropy"] = entropy
    model.tower_stats["pi_err"] = coma_pi_err
    model.tower_stats["value_err"] = value_err

    return total_loss


def loss_and_entropy_stats(policy: Policy,
                           train_batch: SampleBatch) -> Dict[str, TensorType]:
    return {
        "policy_entropy": torch.mean(
            torch.stack(policy.get_tower_stats("entropy"))),
        "policy_loss": torch.mean(
            torch.stack(policy.get_tower_stats("pi_err"))),
        "vf_loss": torch.mean(
            torch.stack(policy.get_tower_stats("value_err"))),
    }


def coma_model_value_predictions(
        policy: Policy, input_dict: Dict[str, TensorType], state_batches,
        model: ModelV2,
        action_dist: ActionDistribution) -> Dict[str, TensorType]:
    return {SampleBatch.VF_PREDS: model.value_function()}


def torch_optimizer(policy: Policy,
                    config: TrainerConfigDict) -> LocalOptimizer:
    return torch.optim.Adam(policy.model.parameters(), lr=config["lr"])


def setup_mixins(policy: Policy, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 config: TrainerConfigDict) -> None:
    """Call all mixin classes' constructors before PPOPolicy initialization.

    Args:
        policy (Policy): The Policy object.
        obs_space (gym.spaces.Space): The Policy's observation space.
        action_space (gym.spaces.Space): The Policy's action space.
        config (TrainerConfigDict): The Policy's config.
    """
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)


def centralized_critic_postprocessing_coma(policy,
                                           sample_batch,
                                           other_agent_batches=None,
                                           episode=None):
    pytorch = policy.config["framework"] == "torch"
    n_agents = policy.config["model"]["custom_model_config"]["agent_num"]
    raw_opponent_agents_num = n_agents - 1
    action_dim = policy.action_space.n
    if (pytorch and hasattr(policy, "compute_central_vf")) or \
            (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None
        opponent_batch_list = list(other_agent_batches.values())
        avail_opponent_agents_num = len(opponent_batch_list)
        agent_index_padding = {i for i in range(n_agents)}
        agent_index_exist = set()

        current_agent_index = sample_batch["agent_index"][0]
        agent_index_padding.remove(current_agent_index)

        for key in other_agent_batches.keys():
            agent_index_padding.remove(int(key[-1]))
            agent_index_exist.add(int(key[-1]))

        raw_opponent_batch = [opponent_batch_list[i][1] for i in range(avail_opponent_agents_num)]
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

        if raw_opponent_agents_num != avail_opponent_agents_num:
            if len(opponent_batch) == 0:
                sample_batch["opponent_obs"] = np.stack([np.zeros_like(sample_batch["obs"]) for _ in range(
                    raw_opponent_agents_num)], 1)
                sample_batch["opponent_action"] = np.stack([np.zeros_like(sample_batch["actions"]) - 1 for _ in range(
                    raw_opponent_agents_num)], 1)  # -1 for fake action
            else:
                agent_index_exist = list(agent_index_exist)
                agent_index_padding = list(agent_index_padding)
                opp_index = agent_index_exist + agent_index_padding
                opp_index.sort()
                exist_agent_count = 0
                opponent_obs_list = []
                opponent_action_list = []

                for agent_index in opp_index:
                    if agent_index in agent_index_exist:
                        opponent_obs_list.append(opponent_batch[exist_agent_count]["obs"])
                        opponent_action_list.append(opponent_batch[exist_agent_count]["actions"])
                        exist_agent_count += 1
                    else:
                        opponent_obs_list.append(np.zeros_like(sample_batch["obs"]))
                        opponent_action_list.append(np.zeros_like(sample_batch["actions"]))

                sample_batch["opponent_obs"] = np.stack(opponent_obs_list, 1)
                sample_batch["opponent_action"] = np.stack(opponent_action_list, 1)

        else:
            sample_batch["opponent_obs"] = np.stack(
                [opponent_batch[i]["obs"] for i in range(raw_opponent_agents_num)], 1)
            sample_batch["opponent_action"] = np.stack(
                [opponent_batch[i]["actions"] for i in range(raw_opponent_agents_num)], 1)

        # overwrite default VF prediction with the central VF
        if policy.config['framework'] == "torch":
            sample_batch["vf_preds"] = policy.compute_central_vf(
                convert_to_torch_tensor(
                    sample_batch["obs"], policy.device),
                convert_to_torch_tensor(
                    sample_batch["opponent_obs"], policy.device),
                convert_to_torch_tensor(
                    sample_batch["opponent_action"], policy.device),
            ).cpu().detach().numpy().mean(1)
        else:
            sample_batch["vf_preds"] = policy.compute_central_vf(
                sample_batch["obs"], sample_batch["opponent_obs"],
                sample_batch["opponent_action"])
    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch["opponent_obs"] = np.zeros(
            (sample_batch["obs"].shape[0], raw_opponent_agents_num, sample_batch["obs"].shape[1]),
            dtype=sample_batch["obs"].dtype)
        sample_batch["opponent_action"] = np.zeros_like(
            sample_batch["actions"])
        sample_batch["opponent_action"] = np.zeros(
            (sample_batch["actions"].shape[0], raw_opponent_agents_num),
            dtype=sample_batch["actions"].dtype)
        sample_batch["vf_preds"] = np.zeros_like(
            sample_batch["rewards"], dtype=np.float32)

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch["vf_preds"][-1]

    sample_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])

    return sample_batch


# based on a3c torch policy
COMATorchPolicy = build_policy_class(
    name="COMATorchPolicy",
    framework="torch",
    get_default_config=lambda: ray.rllib.agents.a3c.a3c.DEFAULT_CONFIG,
    loss_fn=coma_loss,
    stats_fn=loss_and_entropy_stats,
    postprocess_fn=centralized_critic_postprocessing_coma,
    extra_action_out_fn=coma_model_value_predictions,  # may have bug as we have action dim vf function output in coma
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=torch_optimizer,
    before_loss_init=setup_mixins,
    mixins=[ValueNetworkMixin],
)
