import logging
from typing import Optional, Type
import gym
from typing import Tuple
import numpy as np

import ray
from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDeterministic, TorchDirichlet
from ray.rllib.models.tf.tf_action_dist import Deterministic
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog, MODEL_DEFAULTS
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.torch_ops import huber_loss, l2_loss, sequence_mask
from ray.rllib.utils.typing import TrainerConfigDict, TensorType, ModelInputDict
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_DEFAULT_CONFIG
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import TrainerConfigDict

from MaMujoco.model.torch_rnnddpg import RNNDDPGTorchModel

torch, nn = try_import_torch()

def build_rnnddpg_models(policy, observation_space, action_space, config):
    num_outputs = int(np.product(observation_space.shape))

    default_model_cls = RNNDDPGTorchModel

    policy_model_config = MODEL_DEFAULTS.copy()
    policy_model_config.update(config["policy_model"])
    q_model_config = MODEL_DEFAULTS.copy()
    q_model_config.update(config["Q_model"])

    policy.model = ModelCatalog.get_model_v2(
        obs_space=observation_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        default_model=default_model_cls,
        name="rnnddpg_model",
        policy_model_config=policy_model_config,
        q_model_config=q_model_config,
        twin_q=config["twin_q"],
        add_layer_norm=(policy.config["exploration_config"].get("type") ==
                        "ParameterNoise"),
    )

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=observation_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        default_model=default_model_cls,
        name="rnnddpg_model",
        policy_model_config=policy_model_config,
        q_model_config=q_model_config,
        twin_q=config["twin_q"],
        add_layer_norm=(policy.config["exploration_config"].get("type") ==
                        "ParameterNoise"),
    )

    return policy.model


def build_rnnddpg_models_and_action_dist(
        policy: Policy, obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: TrainerConfigDict) -> Tuple[ModelV2, ActionDistribution]:
    model = build_rnnddpg_models(policy, obs_space, action_space, config)

    assert model.get_initial_state() != [], \
        "RNNDDPG requires its model to be a recurrent one!"

    if isinstance(action_space, Simplex):
        return model, TorchDirichlet
    else:
        return model, TorchDeterministic


def action_distribution_fn(policy: Policy,
                           model: ModelV2,
                           input_dict: ModelInputDict,
                           *,
                           explore=True,
                           is_training=False,
                           state_batches=None,
                           seq_lens=None,
                           prev_action_batch=None,
                           prev_reward_batch=None,
                           timestep=None,
                           **kwargs):

    # modify 输入输出要修改
    model_out, state_in = model(input_dict, state_batches, seq_lens)

    states_in = model.select_state(state_in, ["policy", "q", "twin_q"])

    distribution_inputs, policy_state_out = \
        model.get_policy_output(model_out, states_in["policy"], seq_lens)
    _, q_state_out = model.get_q_values(model_out, states_in["q"], seq_lens)
    if model.twin_q_model:
        _, twin_q_state_out = \
            model.get_twin_q_values(model_out, states_in["twin_q"], seq_lens)
    else:
        twin_q_state_out = []

    states_out = policy_state_out + q_state_out + twin_q_state_out

    return distribution_inputs, (TorchDeterministic
                                 if policy.config["framework"] == "torch" else
                                 Deterministic), states_out


def ddpg_actor_critic_loss(policy: Policy, model: ModelV2, _,
                           train_batch: SampleBatch) -> TensorType:

    target_model = policy.target_models[model]

    i = 0
    state_batches = []
    while "state_in_{}".format(i) in train_batch:
        state_batches.append(train_batch["state_in_{}".format(i)])
        i += 1
    assert state_batches
    seq_lens = train_batch.get(SampleBatch.SEQ_LENS)

    twin_q = policy.config["twin_q"]
    gamma = policy.config["gamma"]
    n_step = policy.config["n_step"]
    use_huber = policy.config["use_huber"]
    huber_threshold = policy.config["huber_threshold"]
    l2_reg = policy.config["l2_reg"]

    input_dict = {
        "obs": train_batch[SampleBatch.CUR_OBS],
        "is_training": True,
        "prev_actions": train_batch[SampleBatch.PREV_ACTIONS],
        "prev_rewards": train_batch[SampleBatch.PREV_REWARDS],
    }
    model_out_t, state_in_t = model(input_dict, state_batches, seq_lens)
    states_in_t = model.select_state(state_in_t, ["policy", "q", "twin_q"])

    input_dict_next = {
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": True,
        "prev_actions": train_batch[SampleBatch.ACTIONS],
        "prev_rewards": train_batch[SampleBatch.REWARDS],
    }
    model_out_tp1, state_in_tp1 = model(
        input_dict_next, state_batches, seq_lens)
    states_in_tp1 = model.select_state(state_in_tp1, ["policy", "q", "twin_q"])

    target_model_out_tp1, target_state_in_tp1 = target_model(
        input_dict_next, state_batches, seq_lens)
    target_states_in_tp1 = target_model.select_state(state_in_tp1,
                                                     ["policy", "q", "twin_q"])

    # Policy network evaluation.
    # prev_update_ops = set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS))
    policy_t = model.get_policy_output(
        model_out_t, states_in_t["policy"], seq_lens)[0]  # TODO
    # policy_batchnorm_update_ops = list(
    #    set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS)) - prev_update_ops)

    policy_tp1 = target_model.get_policy_output(
        target_model_out_tp1, target_states_in_tp1["policy"], seq_lens)[0]  # TODO

    # Action outputs.
    if policy.config["smooth_target_policy"]:
        target_noise_clip = policy.config["target_noise_clip"]
        clipped_normal_sample = torch.clamp(
            torch.normal(
                mean=torch.zeros(policy_tp1.size()),
                std=policy.config["target_noise"]).to(policy_tp1.device),
            -target_noise_clip, target_noise_clip)

        policy_tp1_smoothed = torch.min(
            torch.max(
                policy_tp1 + clipped_normal_sample,
                torch.tensor(
                    policy.action_space.low,
                    dtype=torch.float32,
                    device=policy_tp1.device)),
            torch.tensor(
                policy.action_space.high,
                dtype=torch.float32,
                device=policy_tp1.device))
    else:
        # No smoothing, just use deterministic actions.
        policy_tp1_smoothed = policy_tp1

    # Q-net(s) evaluation.
    # prev_update_ops = set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS))
    # Q-values for given actions & observations in given current
    q_t = model.get_q_values(
        model_out_t, states_in_t["q"], seq_lens, train_batch[SampleBatch.ACTIONS])[0]

    # Q-values for current policy (no noise) in given current state
    q_t_det_policy = model.get_q_values(
        model_out_t, states_in_t["q"], seq_lens, policy_t)[0]

    actor_loss = -torch.mean(q_t_det_policy)

    if twin_q:
        twin_q_t = model.get_twin_q_values(model_out_t, states_in_t["twin_q"], seq_lens,
                                           train_batch[SampleBatch.ACTIONS])
    # q_batchnorm_update_ops = list(
    #     set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS)) - prev_update_ops)

    # Target q-net(s) evaluation.
    q_tp1 = target_model.get_q_values(
        target_model_out_tp1, target_states_in_tp1["q"], seq_lens, policy_tp1_smoothed)[0]

    if twin_q:
        twin_q_tp1 = target_model.get_twin_q_values(target_model_out_tp1, target_states_in_tp1["twin_q"], seq_lens,
                                                    policy_tp1_smoothed)

    q_t_selected = torch.squeeze(q_t, axis=len(q_t.shape) - 1)
    if twin_q:
        twin_q_t_selected = torch.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
        q_tp1 = torch.min(q_tp1, twin_q_tp1)

    q_tp1_best = torch.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
    q_tp1_best_masked = \
        (1.0 - train_batch[SampleBatch.DONES].float()) * \
        q_tp1_best

    # Compute RHS of bellman equation.
    q_t_selected_target = (train_batch[SampleBatch.REWARDS] +
                           gamma**n_step * q_tp1_best_masked).detach()

    # BURNIN #
    B = state_batches[0].shape[0]
    T = q_t_selected.shape[0] // B
    seq_mask = sequence_mask(train_batch[SampleBatch.SEQ_LENS], T)
    # Mask away also the burn-in sequence at the beginning.
    burn_in = policy.config["burn_in"]
    if burn_in > 0 and burn_in < T:
        seq_mask[:, :burn_in] = False

    seq_mask = seq_mask.reshape(-1)
    num_valid = torch.sum(seq_mask)

    def reduce_mean_valid(t):
        return torch.sum(t[seq_mask]) / num_valid

    # Compute the error (potentially clipped).
    if twin_q:
        td_error = q_t_selected - q_t_selected_target
        td_error = td_error * seq_mask
        twin_td_error = twin_q_t_selected - q_t_selected_target
        if use_huber:
            errors = huber_loss(td_error, huber_threshold) \
                + huber_loss(twin_td_error, huber_threshold)
        else:
            errors = 0.5 * \
                (torch.pow(td_error, 2.0) + torch.pow(twin_td_error, 2.0))
    else:
        td_error = q_t_selected - q_t_selected_target
        td_error = td_error * seq_mask
        if use_huber:
            errors = huber_loss(td_error, huber_threshold)
        else:
            errors = 0.5 * torch.pow(td_error, 2.0)

    critic_loss = torch.mean(train_batch[PRIO_WEIGHTS] * errors)

    # Add l2-regularization if required.
    if l2_reg is not None:
        for name, var in model.policy_variables(as_dict=True).items():
            if "bias" not in name:
                actor_loss += (l2_reg * l2_loss(var))
        for name, var in model.q_variables(as_dict=True).items():
            if "bias" not in name:
                critic_loss += (l2_reg * l2_loss(var))

    # Model self-supervised losses.
    if policy.config["use_state_preprocessor"]:
        # Expand input_dict in case custom_loss' need them.
        input_dict[SampleBatch.ACTIONS] = train_batch[SampleBatch.ACTIONS]
        input_dict[SampleBatch.REWARDS] = train_batch[SampleBatch.REWARDS]
        input_dict[SampleBatch.DONES] = train_batch[SampleBatch.DONES]
        input_dict[SampleBatch.NEXT_OBS] = train_batch[SampleBatch.NEXT_OBS]
        [actor_loss, critic_loss] = model.custom_loss(
            [actor_loss, critic_loss], input_dict)

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["q_t"] = q_t * seq_mask[..., None]
    model.tower_stats["actor_loss"] = actor_loss
    model.tower_stats["critic_loss"] = critic_loss
    # TD-error tensor in final stats
    # will be concatenated and retrieved for each individual batch item.
    model.tower_stats["td_error"] = td_error

    # Return two loss terms (corresponding to the two optimizers, we create).
    return actor_loss, critic_loss

logger = logging.getLogger(__name__)


RNNDDPG_DEFAULT_CONFIG = DDPGTrainer.merge_trainer_configs(
    DDPG_DEFAULT_CONFIG,
    {
        # Batch mode (see common config)
        "batch_mode": "complete_episodes",
        # If True, assume a zero-initialized state input (no matter where in
        # the episode the sequence is located).
        # If False, store the initial states along with each SampleBatch, use
        # it (as initial state when running through the network for training),
        # and update that initial state during training (from the internal
        # state outputs of the immediately preceding sequence).
        "zero_init_states": True,
        # If > 0, use the `burn_in` first steps of each replay-sampled sequence
        # (starting either from all 0.0-values if `zero_init_state=True` or
        # from the already stored values) to calculate an even more accurate
        # initial states for the actual sequence (starting after this burn-in
        # window). In the burn-in case, the actual length of the sequence
        # used for loss calculation is `n - burn_in` time steps
        # (n=LSTM’s/attention net’s max_seq_len).
        "burn_in": 0,
        # Set automatically: The number of contiguous environment steps to
        # replay at once. Will be calculated via
        # model->max_seq_len + burn_in.
        # Do not set this to any valid value!
        "replay_sequence_length": -1,

        "Q_model": None,
        "policy_model": None
    },
    _allow_unknown_configs=True,
)


def validate_config(config: TrainerConfigDict) -> None:
    if config["replay_sequence_length"] != -1:
        raise ValueError(
            "`replay_sequence_length` is calculated automatically to be "
            "model->max_seq_len + burn_in!")
    # Add the `burn_in` to the Model's max_seq_len.
    # Set the replay sequence length to the max_seq_len of the model.
    config["replay_sequence_length"] = \
        config["burn_in"] + config["model"]["max_seq_len"]

