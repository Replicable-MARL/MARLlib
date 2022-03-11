"""PyTorch policy class used for R2D2."""

from ray.rllib.agents.dqn.dqn_tf_policy import (PRIO_WEIGHTS,
                                                postprocess_nstep_and_prio)
from ray.rllib.agents.dqn.r2d2_torch_policy import *
from typing import Dict, List, Tuple

import ray
from ray.rllib.agents.dqn.dqn_tf_policy import (
    PRIO_WEIGHTS, Q_SCOPE, Q_TARGET_SCOPE, postprocess_nstep_and_prio)
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.agents.dqn.simple_q_torch_policy import TargetNetworkMixin
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import (TorchCategorical,
                                                      TorchDistributionWrapper)
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.exploration.parameter_noise import ParameterNoise
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    concat_multi_gpu_td_errors, FLOAT_MIN, huber_loss, \
    reduce_mean_ignore_inf, softmax_cross_entropy_with_logits
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
import gym

from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch()
F = None
if nn:
    F = nn.functional


class DQNTorchModel_Support_Mask(DQNTorchModel):

    def get_q_value_distributions(self, model_out):
        """Returns distributional values for Q(s, a) given a state embedding.

        Override this in your custom model to customize the Q output head.

        Args:
            model_out (Tensor): Embedding from the model layers.

        Returns:
            (action_scores, logits, dist) if num_atoms == 1, otherwise
            (action_scores, z, support_logits_per_action, logits, dist)
        """

        # first is feature; second is action mask -inf
        action_scores = self.advantage_module(model_out[0])
        action_scores = action_scores + model_out[1]

        if self.num_atoms > 1:
            # Distributional Q-learning uses a discrete support z
            # to represent the action value distribution
            z = torch.range(
                0.0, self.num_atoms - 1,
                dtype=torch.float32).to(action_scores.device)
            z = self.v_min + \
                z * (self.v_max - self.v_min) / float(self.num_atoms - 1)

            support_logits_per_action = torch.reshape(
                action_scores, shape=(-1, self.action_space.n, self.num_atoms))
            support_prob_per_action = nn.functional.softmax(
                support_logits_per_action, dim=-1)
            action_scores = torch.sum(z * support_prob_per_action, dim=-1)
            logits = support_logits_per_action
            probs = support_prob_per_action
            return action_scores, z, support_logits_per_action, logits, probs
        else:
            logits = torch.unsqueeze(torch.ones_like(action_scores), -1)
            return action_scores, logits, logits

    def get_state_value(self, model_out):
        """Returns the state value prediction for the given state embedding."""

        return self.value_module(model_out[0])


def build_q_model_and_distribution_with_mask(
        policy: Policy, obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: TrainerConfigDict) -> Tuple[ModelV2, TorchDistributionWrapper]:
    """Build q_model and target_model for DQN

    Args:
        policy (Policy): The policy, which will use the model for optimization.
        obs_space (gym.spaces.Space): The policy's observation space.
        action_space (gym.spaces.Space): The policy's action space.
        config (TrainerConfigDict):

    Returns:
        (q_model, TorchCategorical)
            Note: The target q model will not be returned, just assigned to
            `policy.target_model`.
    """
    if not isinstance(action_space, gym.spaces.Discrete):
        raise UnsupportedSpaceException(
            "Action space {} is not supported for DQN.".format(action_space))

    if config["hiddens"]:
        # try to infer the last layer size, otherwise fall back to 256
        num_outputs = ([256] + list(config["model"]["fcnet_hiddens"]))[-1]
        config["model"]["no_final_linear"] = True
    else:
        num_outputs = action_space.n

    # TODO(sven): Move option to add LayerNorm after each Dense
    #  generically into ModelCatalog.
    add_layer_norm = (
            isinstance(getattr(policy, "exploration", None), ParameterNoise)
            or config["exploration_config"]["type"] == "ParameterNoise")

    model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework="torch",
        model_interface=DQNTorchModel_Support_Mask,
        name=Q_SCOPE,
        q_hiddens=config["hiddens"],
        dueling=config["dueling"],
        num_atoms=config["num_atoms"],
        use_noisy=config["noisy"],
        v_min=config["v_min"],
        v_max=config["v_max"],
        sigma0=config["sigma0"],
        # TODO(sven): Move option to add LayerNorm after each Dense
        #  generically into ModelCatalog.
        add_layer_norm=add_layer_norm)

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework="torch",
        model_interface=DQNTorchModel_Support_Mask,
        name=Q_TARGET_SCOPE,
        q_hiddens=config["hiddens"],
        dueling=config["dueling"],
        num_atoms=config["num_atoms"],
        use_noisy=config["noisy"],
        v_min=config["v_min"],
        v_max=config["v_max"],
        sigma0=config["sigma0"],
        # TODO(sven): Move option to add LayerNorm after each Dense
        #  generically into ModelCatalog.
        add_layer_norm=add_layer_norm)

    return model, TorchCategorical


def build_r2d2_model_and_distribution_with_mask(
        policy: Policy, obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: TrainerConfigDict) -> \
        Tuple[ModelV2, TorchDistributionWrapper]:
    """Build q_model and target_model for DQN

    Args:
        policy (Policy): The policy, which will use the model for optimization.
        obs_space (gym.spaces.Space): The policy's observation space.
        action_space (gym.spaces.Space): The policy's action space.
        config (TrainerConfigDict):

    Returns:
        (q_model, TorchCategorical)
            Note: The target q model will not be returned, just assigned to
            `policy.target_model`.
    """

    # Create the policy's models and action dist class.
    model, distribution_cls = build_q_model_and_distribution_with_mask(
        policy, obs_space, action_space, config)

    # Assert correct model type by checking the init state to be present.
    # For attention nets: These don't necessarily publish their init state via
    # Model.get_initial_state, but may only use the trajectory view API
    # (view_requirements).
    assert (model.get_initial_state() != [] or
            model.view_requirements.get("state_in_0") is not None), \
        "R2D2 requires its model to be a recurrent one! Try using " \
        "`model.use_lstm` or `model.use_attention` in your config " \
        "to auto-wrap your model with an LSTM- or attention net."

    return model, distribution_cls


R2D2WithMaskPolicy = build_policy_class(
    name="R2D2WithMaskPolicy",
    framework="torch",
    loss_fn=r2d2_loss,
    get_default_config=lambda: ray.rllib.agents.dqn.r2d2.DEFAULT_CONFIG,
    make_model_and_action_dist=build_r2d2_model_and_distribution_with_mask,
    action_distribution_fn=get_distribution_inputs_and_class,
    stats_fn=build_q_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    optimizer_fn=adam_optimizer,
    extra_grad_process_fn=grad_process_and_td_error_fn,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    extra_action_out_fn=extra_action_out_fn,
    before_init=setup_early_mixins,
    before_loss_init=before_loss_init,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ])
