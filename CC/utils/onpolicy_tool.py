from ray.rllib.utils.torch_ops import apply_grad_clipping, sequence_mask
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import TrainerConfigDict, TensorType, \
    LocalOptimizer, GradInfoDict
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy, actor_critic_loss
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG, A2CTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, ValueNetworkMixin, KLCoeffMixin, ppo_surrogate_loss
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
import numpy as np
from typing import Dict, Tuple

torch, nn = try_import_torch()


##############
### COMMON ###
##############

def get_dim(a):
    dim = 1
    for i in a:
        dim *= i
    return dim


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        self.compute_central_vf = self.model.central_value_function


def centralized_critic_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):
    custom_config = policy.config["model"]["custom_model_config"]
    pytorch = custom_config["framework"] == "torch"
    obs_dim = get_dim(custom_config["space_obs"]["obs"].shape)
    algorithm = custom_config["algorithm"]
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

    if (pytorch and hasattr(policy, "compute_central_vf")) or \
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

        sample_batch["opponent_action"] = np.stack([opponent_batch[i]["actions"] for i in range(opponent_agents_num)],
                                                   1)
        if algorithm == "coma":
            sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
                convert_to_torch_tensor(
                    sample_batch["state"], policy.device),
                convert_to_torch_tensor(
                    sample_batch["opponent_action"], policy.device),
            ) \
                .cpu().detach().numpy().mean(1)
            sample_batch[SampleBatch.VF_PREDS] = np.take(sample_batch[SampleBatch.VF_PREDS],
                                                         np.expand_dims(sample_batch["actions"], axis=1)).squeeze(
                axis=1)

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
        sample_batch["opponent_action"] = np.stack(
            [np.zeros_like(sample_batch["actions"], dtype=sample_batch["actions"].dtype) for _ in
             range(opponent_agents_num)], axis=1)
        # sample_batch["opponent_action"] = np.zeros(
        #     (sample_batch["actions"].shape[0], opponent_agents_num),
        #     dtype=sample_batch["actions"].dtype)

        if algorithm == "coma":
            sample_batch[SampleBatch.VF_PREDS] = np.take(sample_batch[SampleBatch.VF_PREDS],
                                                         np.expand_dims(sample_batch["actions"], axis=1)).squeeze(
                axis=1)

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch


#############
### MAA2C ###
#############

# Copied from A2C but optimizing the central value function.
def central_critic_a2c_loss(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = actor_critic_loss

    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(train_batch["state"],
                                                                       train_batch["opponent_action"])

    # recording data
    policy._central_value_out = model.value_function()

    loss = func(policy, model, dist_class, train_batch)
    model.value_function = vf_saved

    return loss


MAA2CTorchPolicy = A3CTorchPolicy.with_updates(
    name="MAA2CTorchPolicy",
    get_default_config=lambda: A2C_CONFIG,
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=central_critic_a2c_loss,
    mixins=[
        CentralizedValueMixin
    ])


def get_policy_class_maa2c(config_):
    if config_["framework"] == "torch":
        return MAA2CTorchPolicy


MAA2CTrainer = A2CTrainer.with_updates(
    name="MAA2CTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_maa2c,
)


#############
### MAPPO ###
#############

# Copied from PPO but optimizing the central value function.
def setup_torch_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTorchPolicy  (w/o ValueNetworkMixin).
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def central_critic_ppo_loss(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = ppo_surrogate_loss

    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(
        train_batch["state"], train_batch["opponent_action"])

    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)

    model.value_function = vf_saved

    return loss


MAPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="MAPPOTorchPolicy",
    get_default_config=lambda: PPO_CONFIG,
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=central_critic_ppo_loss,
    before_init=setup_torch_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        CentralizedValueMixin
    ])


def get_policy_class_mappo(config_):
    if config_["framework"] == "torch":
        return MAPPOTorchPolicy


MAPPOTrainer = PPOTrainer.with_updates(
    name="MAPPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_mappo,
)


############
### COMA ###
############

def central_critic_coma_loss(policy: Policy, model: ModelV2,
                             dist_class: ActionDistribution,
                             train_batch: SampleBatch) -> TensorType:
    CentralizedValueMixin.__init__(policy)
    logits, _ = model.from_batch(train_batch)
    values = model.central_value_function(convert_to_torch_tensor(
        train_batch["state"], policy.device),
        convert_to_torch_tensor(
            train_batch["opponent_action"], policy.device))
    pi = torch.nn.functional.softmax(logits, dim=-1)

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
    advantages = (select_action_Q_value - torch.sum(values * pi, dim=1)).detach()
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


def coma_model_value_predictions(
        policy: Policy, input_dict: Dict[str, TensorType], state_batches,
        model: ModelV2,
        action_dist: ActionDistribution) -> Dict[str, TensorType]:
    return {SampleBatch.VF_PREDS: model.value_function()}


COMATorchPolicy = A3CTorchPolicy.with_updates(
    name="COMATorchPolicy",
    get_default_config=lambda: A2C_CONFIG,
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=central_critic_coma_loss,
    extra_action_out_fn=coma_model_value_predictions,
    mixins=[
        CentralizedValueMixin
    ])


def get_policy_class_coma(config_):
    if config_["framework"] == "torch":
        return COMATorchPolicy


COMATrainer = A2CTrainer.with_updates(
    name="COMATrainer",
    default_policy=None,
    get_policy_class=get_policy_class_coma,
)
