"""
Implement TRPO and HATRPO in Ray Rllib
__author__: minquan
__data__: May-15
"""

import logging
from typing import Dict, List, Type, Union, Tuple
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    explained_variance, sequence_mask
import numpy as np
from ray.rllib.evaluation.postprocessing import discount_cumsum, Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf, try_import_torch, get_variable
from torch import nn
from marl.algos.utils.valuenorm import ValueNorm
from ray.rllib.utils.typing import TrainerConfigDict, TensorType, \
    LocalOptimizer
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, ValueNetworkMixin, KLCoeffMixin
from ray.rllib.utils.torch_ops import apply_grad_clipping
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from functools import partial
from marl.algos.utils.setup_utils import setup_torch_mixins
from marl.algos.utils.get_hetero_info import (
    get_global_name,
    contain_global_obs,
    GLOBAL_MODEL,
    GLOBAL_IS_TRAINING,
    hatrpo_post_process,
    value_normalizer,
    STATE,
)

from marl.algos.utils.trust_regions import update_model_use_trust_region

from ray.rllib.examples.centralized_critic import CentralizedValueMixin
import ctypes
from icecream import ic

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


logger = logging.getLogger(__name__)


def recovery_obj(_id):
    return ctypes.cast(_id, ctypes.py_object).value


def hatrpo_loss_fn(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for TRPO
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
        opp_action_in_cc = policy.config["model"]["custom_model_config"]["opp_action_in_cc"]
        model.value_function = lambda: policy.model.central_value_function(
            train_batch[STATE],
            train_batch[get_global_name(SampleBatch.ACTIONS)]
            if opp_action_in_cc else None
        )
        # model.value_function = lambda: policy.model.central_value_function(
        #     train_batch[SampleBatch.OBS], train_batch[get_global_name(SampleBatch.ACTIONS)]
        # )

    want_hatrpo_but_other_opponent_not_initialized = (GLOBAL_MODEL not in train_batch)

    if want_hatrpo_but_other_opponent_not_initialized:
        policy_loss_for_rllib, action_kl = update_model_use_trust_region(
            model=model,
            train_batch=train_batch,
            advantages=train_batch[Postprocessing.ADVANTAGES],
            # pre_action_dist=train_batch[SampleBatch.]
            actions=train_batch[SampleBatch.ACTIONS],
            action_logp=train_batch[SampleBatch.ACTION_LOGP],
            action_dist_inputs=train_batch[SampleBatch.ACTION_DIST_INPUTS],
            dist_class=dist_class,
            mean_fn=reduce_mean_valid,
        )
    else:
        # for _ in range(10):
        #     print('step into HATRPO')
        m_advantage = train_batch[Postprocessing.ADVANTAGES]
        agents_num = train_batch[get_global_name(SampleBatch.ACTION_DIST_INPUTS)].shape[1] + 1
        random_indices = np.random.permutation(range(agents_num))

        policy_losses = []
        kl_losses = []

        def is_current_agent(i): return i == agents_num - 1

        for agent_id in random_indices:
            if is_current_agent(agent_id):
                current_model = model
                logits, state = model(train_batch)
                current_action_dist = dist_class(logits, model)
                old_action_log_dist = train_batch[SampleBatch.ACTION_LOGP]
                actions = train_batch[SampleBatch.ACTIONS]
                train_batch_for_trpo_update = train_batch
                action_dist_input = train_batch[SampleBatch.ACTION_DIST_INPUTS]
            else:
                current_model = recovery_obj(int(train_batch[GLOBAL_MODEL][agent_id]))
                # train_batch_for_trpo_update = train_batch[GLOBAL_TRAIN_BATCH][agent_id]
                current_action_logits = train_batch[get_global_name(SampleBatch.ACTION_DIST_INPUTS)][:, agent_id, :].detach()
                current_action_dist = dist_class(current_action_logits, None)
                # current_action_dist = train_batch[GLOBAL_MODEL_LOGITS][:, agent_id, :]
                old_action_log_dist = train_batch[get_global_name(SampleBatch.ACTION_LOGP)][:, agent_id].detach()
                actions = train_batch[get_global_name(SampleBatch.ACTIONS)][:, agent_id].detach()
                # actions = train_batch_for_trpo_update[SampleBatch.ACTIONS]
                action_dist_input = train_batch[get_global_name(SampleBatch.ACTION_DIST_INPUTS)][:, agent_id].detach()

                train_batch_for_trpo_update = SampleBatch(
                    obs=train_batch[get_global_name(SampleBatch.OBS)][:, agent_id],
                    seq_lens=train_batch[get_global_name(SampleBatch.SEQ_LENS)][:, agent_id]
                )

                train_batch_for_trpo_update.is_training = bool(train_batch[GLOBAL_IS_TRAINING][agent_id])

                i = 0

                def _state_name(i): return f'state_in_{i}'

                while _state_name(i) in train_batch:
                    train_batch_for_trpo_update[_state_name(i)] = train_batch[get_global_name(_state_name(i))][:, agent_id]
                    i += 1

                # train_batch_for_trpo_update['state_in_0'] = train_batch[GLOBAL_STATE][:, agent_id].reshape(
                #     train_batch['state_in_0'].shape
                # )
                # for i, s in enumerate(train_batch[GLOBAL_STATE][:, agent_id]):
                #     state_name = f'state_in_{i}'
                #     state = s.reshape(train_batch[state_name].shape)
                #     train_batch_for_trpo_update[state_name] = state

            importance_sampling = torch.exp(current_action_dist.logp(actions) - old_action_log_dist)

            ic(train_batch_for_trpo_update['obs'].shape)
            ic(train_batch_for_trpo_update['state_in_0'].shape)

            kl_loss, policy_loss = update_model_use_trust_region(
                    model=current_model,
                    train_batch=train_batch_for_trpo_update,
                    advantages=m_advantage,
                    actions=actions,
                    action_logp=old_action_log_dist,
                    action_dist_inputs=action_dist_input,
                    mean_fn=reduce_mean_valid,
                    dist_class=dist_class,
            )
            kl_losses.append(kl_loss)
            policy_losses.append(policy_loss)

            m_advantage = importance_sampling * m_advantage

            # sub_losses.append(sub_loss)

        policy_loss_for_rllib = torch.mean(torch.stack(policy_losses, axis=1), axis=1)
        action_kl = torch.mean(torch.stack(kl_losses, axis=1), axis=1)

    curr_entropy = curr_action_dist.entropy()

    # Compute a value function loss.
    # if policy.model.model_config['custom_model_config']['normal_value']:
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

    total_loss = reduce_mean_valid(-policy_loss_for_rllib +
                                   policy.kl_coeff * action_kl +
                                   policy.config["vf_loss_coeff"] * vf_loss -
                                   policy.entropy_coeff * curr_entropy)

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    mean_kl_loss = reduce_mean_valid(action_kl)
    mean_policy_loss = reduce_mean_valid(-policy_loss_for_rllib)
    mean_entropy = reduce_mean_valid(curr_entropy)

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], model.value_function())
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss


HAPTRPOTorchPolicy = lambda _config: PPOTorchPolicy.with_updates(
        name="HAPPOTorchPolicy",
        get_default_config=lambda: _config,
        postprocess_fn=hatrpo_post_process,
        loss_fn=hatrpo_loss_fn,
        before_init=setup_torch_mixins,
        extra_grad_process_fn=apply_grad_clipping,
        mixins=[
            EntropyCoeffSchedule, KLCoeffMixin,
            CentralizedValueMixin, LearningRateSchedule,
        ])


def get_policy_class_hatrpo(config_):
    if config_["framework"] == "torch":
        return HAPTRPOTorchPolicy


HATRPOTrainer = lambda ppo_config: PPOTrainer.with_updates(
    name="#hatrpo-trainer",
    default_policy=None,
    get_policy_class=get_policy_class_hatrpo,
)