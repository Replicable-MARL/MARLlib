"""An example of customizing PPO to leverage a centralized critic.

Here the model and policy are hard-coded to implement a centralized critic
for TwoStepGame, but you can adapt this for your own use cases.

Compared to simply running `rllib/examples/two_step_game.py --run=PPO`,
this centralized critic version reaches vf_explained_variance=1.0 more stably
since it takes into account the opponent actions as well as the policy's.
Note that this is also using two independent policies instead of weight-sharing
with one.

See also: centralized_critic_2.py for a simpler approach that instead
modifies the environment.
"""

from ray.rllib.agents.ppo.ppo_tf_policy import KLCoeffMixin, \
    ppo_surrogate_loss as tf_loss
from ray.rllib.agents.ppo.ppo_torch_policy import KLCoeffMixin as TorchKLCoeffMixin, ppo_surrogate_loss as torch_loss
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import LearningRateSchedule, \
    EntropyCoeffSchedule
from ray.rllib.policy.torch_policy import LearningRateSchedule as TorchLR, \
    EntropyCoeffSchedule as TorchEntropyCoeffSchedule
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from gym.spaces.box import Box
from ray.rllib.evaluation.postprocessing import adjust_nstep
from ray.rllib.utils.numpy import convert_to_numpy
import numpy as np
from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class CentralizedValueMixin:
    """Add method to evaluate the central value function from the model."""

    def __init__(self):
        if self.config["framework"] != "torch":
            self.compute_central_vf = make_tf_callable(self.get_session())(
                self.model.central_value_function)
        else:
            self.compute_central_vf = self.model.central_value_function


# Grabs the opponent obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(policy,
                                      sample_batch,
                                      other_agent_batches=None,
                                      episode=None):
    pytorch = policy.config["framework"] == "torch"
    n_agents = policy.config["model"]["custom_model_config"]["agent_num"]
    opponent_agents_num = n_agents - 1
    continues = True if policy.action_space.__class__ == Box else False
    # action_dim = policy.action_space.n
    if (pytorch and hasattr(policy, "compute_central_vf")) or \
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

        sample_batch["opponent_obs"] = np.stack([opponent_batch[i]["obs"] for i in range(opponent_agents_num)], 1)
        sample_batch["opponent_action"] = np.stack([opponent_batch[i]["actions"] for i in range(opponent_agents_num)],
                                                   1)

        # overwrite default VF prediction with the central VF
        if policy.config['framework'] == "torch":
            sample_batch["vf_preds"] = policy.compute_central_vf(
                convert_to_torch_tensor(
                    sample_batch["obs"], policy.device),
                convert_to_torch_tensor(
                    sample_batch["opponent_obs"], policy.device),
                convert_to_torch_tensor(
                    sample_batch["opponent_action"], policy.device)) \
                .cpu().detach().numpy()
        else:
            sample_batch["vf_preds"] = policy.compute_central_vf(
                sample_batch["obs"], sample_batch["opponent_obs"],
                sample_batch["opponent_action"])
    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch["opponent_obs"] = np.zeros(
            (sample_batch["obs"].shape[0], opponent_agents_num, sample_batch["obs"].shape[1]),
            dtype=sample_batch["obs"].dtype)
        if not continues:
            sample_batch["opponent_action"] = np.zeros(
                (sample_batch["actions"].shape[0], opponent_agents_num),
                dtype=sample_batch["actions"].dtype)
        else:
            sample_batch["opponent_action"] = np.zeros(
                (sample_batch["actions"].shape[0], opponent_agents_num, sample_batch["actions"].shape[1]),
                dtype=sample_batch["actions"].dtype)
        sample_batch["vf_preds"] = np.zeros_like(
            sample_batch["rewards"], dtype=np.float32)

    if "DDPG" in str(policy.__class__): # MADDPG
        ## copied from postprocess_nstep_and_prio in DDPGTorchPolicy
        if policy.config["n_step"] > 1:
            adjust_nstep(policy.config["n_step"], policy.config["gamma"], sample_batch)

        # Create dummy prio-weights (1.0) in case we don't have any in
        # the batch.
        PRIO_WEIGHTS = "weights"
        if PRIO_WEIGHTS not in sample_batch:
            sample_batch[PRIO_WEIGHTS] = np.ones_like(sample_batch[SampleBatch.REWARDS])

        # Prioritize on the worker side.
        if sample_batch.count > 0 and policy.config["worker_side_prioritization"]:
            td_errors = policy.compute_td_error(
                sample_batch[SampleBatch.OBS], sample_batch[SampleBatch.ACTIONS],
                sample_batch[SampleBatch.REWARDS], sample_batch[SampleBatch.NEXT_OBS],
                sample_batch[SampleBatch.DONES], sample_batch[PRIO_WEIGHTS])
            new_priorities = (np.abs(convert_to_numpy(td_errors)) +
                              policy.config["prioritized_replay_eps"])
            sample_batch[PRIO_WEIGHTS] = new_priorities

    else:  # MAAC MAPPO
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


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = tf_loss if not policy.config["framework"] == "torch" else torch_loss

    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(
        train_batch["obs"], train_batch["opponent_obs"],
        train_batch["opponent_action"])

    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)

    model.value_function = vf_saved

    return loss


def setup_tf_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTFPolicy (w/o ValueNetworkMixin).
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def setup_torch_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTorchPolicy  (w/o ValueNetworkMixin).
    TorchKLCoeffMixin.__init__(policy, config)
    TorchEntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                       config["entropy_coeff_schedule"])
    TorchLR.__init__(policy, config["lr"], config["lr_schedule"])


def central_vf_stats_ppo(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy._central_value_out)
    }
