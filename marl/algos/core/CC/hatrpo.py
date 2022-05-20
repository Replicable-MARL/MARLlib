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
    STATE,
    GLOBAL_MODEL_LOGITS,
    GLOBAL_TRAIN_BATCH,
    GLOBAL_MODEL,
    hatrpo_post_process,
    trpo_post_process,
)

from marl.algos.utils.trust_regions import update_model_use_trust_region

from ray.rllib.examples.centralized_critic import CentralizedValueMixin

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


value_normalizer = ValueNorm(1)

logger = logging.getLogger(__name__)


def __trpo_loss(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch, algorithm='TRPO') -> Union[TensorType, List[TensorType]]:
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

    TRPO, HATRPO = 'TRPO', 'HATRPO'

    CentralizedValueMixin.__init__(policy)

    try:
        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)
    except ValueError:
        print('Nan in model()')
        print(logits)
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
            train_batch[STATE], train_batch[get_global_name(SampleBatch.ACTIONS)]
        )

    want_hatrpo_but_no_other_batches_info = algorithm == HATRPO and not contain_global_obs(train_batch)

    if algorithm == TRPO or want_hatrpo_but_no_other_batches_info:
        policy_loss_for_rllib, action_kl = update_model_use_trust_region(
            model=model,
            train_batch=train_batch,
            advantages=train_batch[Postprocessing.ADVANTAGES],
            obs=train_batch[SampleBatch.OBS],
            actions=train_batch[SampleBatch.ACTIONS],
            action_logp=train_batch[SampleBatch.ACTION_LOGP],
            action_dist_inputs=train_batch[SampleBatch.ACTION_DIST_INPUTS],
            dist_class=dist_class,
            mean_fn=reduce_mean_valid,
        )
    elif algorithm == HATRPO:
        for _ in range(10):
            print('step into HATRPO')
        m_advantage = train_batch[Postprocessing.ADVANTAGES]
        agents_num = train_batch[GLOBAL_MODEL_LOGITS].shape[1] + 1
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
            else:
                train_batch_for_trpo_update = train_batch[GLOBAL_TRAIN_BATCH][agent_id]
                current_model = train_batch[GLOBAL_MODEL][agent_id]
                current_action_logits = train_batch[GLOBAL_MODEL_LOGITS][:, agent_id, :].detach()
                current_action_dist = dist_class(current_action_logits, None)
                # current_action_dist = train_batch[GLOBAL_MODEL_LOGITS][:, agent_id, :]
                old_action_log_dist = train_batch[get_global_name(SampleBatch.ACTION_LOGP)][:, agent_id].detach()
                actions = train_batch[get_global_name(SampleBatch.ACTIONS)][:, agent_id, :].detach()
                # actions = train_batch_for_trpo_update[SampleBatch.ACTIONS]

            importance_sampling = torch.exp(current_action_dist.logp(actions) - old_action_log_dist)

            kl_loss, policy_loss = update_model_use_trust_region(
                model=current_model,
                train_batch=train_batch_for_trpo_update,
                advantages=m_advantage,
                actions=actions,
                action_logp=train_batch_for_trpo_update[SampleBatch.ACTION_LOGP],
                action_dist_inputs=train_batch_for_trpo_update[SampleBatch.ACTION_DIST_INPUTS],
                mean_fn=reduce_mean_valid
            )

            m_advantage = importance_sampling * m_advantage

            # sub_losses.append(sub_loss)
            kl_losses.append(kl_loss)
            policy_losses.append(policy_loss)

        policy_loss_for_rllib = torch.mean(torch.stack(policy_losses, axis=1), axis=1)
        action_kl = torch.mean(torch.stack(kl_losses, axis=1), axis=1)

    curr_entropy = curr_action_dist.entropy()

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

    total_loss = reduce_mean_valid(policy_loss_for_rllib +
                                   policy.kl_coeff * action_kl +
                                   policy.config["vf_loss_coeff"] * vf_loss -
                                   policy.entropy_coeff * curr_entropy)

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    mean_kl_loss = reduce_mean_valid(action_kl)
    mean_policy_loss = reduce_mean_valid(policy_loss_for_rllib)
    mean_entropy = reduce_mean_valid(curr_entropy)

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], model.value_function())
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss


HATRPO, TRPO = 'HATRPO', 'TRPO'
hatrpo_loss_fn = partial(__trpo_loss, algorithm=HATRPO)
trpo_loss_fn = partial(__trpo_loss, algorithm=TRPO)


HAPTRPOTorchPolicy = lambda config: PPOTorchPolicy.with_updates(
        name="HAPPOTorchPolicy",
        get_default_config=lambda: config,
        postprocess_fn=hatrpo_post_process,
        loss_fn=hatrpo_loss_fn,
        before_init=setup_torch_mixins,
        # optimizer_fn=make_happo_optimizers,
        extra_grad_process_fn=apply_grad_clipping,
        mixins=[
            EntropyCoeffSchedule, KLCoeffMixin,
            CentralizedValueMixin, LearningRateSchedule,
        ])


TRPOTorchPolicy = lambda config: PPOTorchPolicy.with_updates(
        name="HAPPOTorchPolicy",
        get_default_config=lambda: config,
        postprocess_fn=trpo_post_process,
        loss_fn=hatrpo_loss_fn,
        before_init=setup_torch_mixins,
        # optimizer_fn=make_happo_optimizers,
        extra_grad_process_fn=apply_grad_clipping,
        mixins=[
            EntropyCoeffSchedule, KLCoeffMixin,
            CentralizedValueMixin, LearningRateSchedule,
        ])


def get_policy_class(ppo_config, default_policy):
    def _get_policy_class(config_):
        if config_["framework"] == "torch":
            return default_policy(ppo_config)
    return _get_policy_class()


HATRPOTrainer = lambda ppo_config: PPOTrainer.with_updates(
    name="#04-08-8-Worker-add-value-normal-critical-lr-1e-2#-HAPPOTrainer-with-local-mode-False",
    default_policy=HATRPOTrainer(ppo_config),
    get_policy_class=get_policy_class(ppo_config, default_policy=HAPTRPOTorchPolicy),
)


TRPOTrainer = lambda ppo_config: PPOTrainer.with_updates(
    name="#04-08-8-Worker-add-value-normal-critical-lr-1e-2#-HAPPOTrainer-with-local-mode-False",
    default_policy=TRPOTorchPolicy(ppo_config),
    get_policy_class=get_policy_class(ppo_config, default_policy=TRPOTorchPolicy),
)
