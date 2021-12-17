from ray.rllib.agents.a3c.a3c_tf_policy import actor_critic_loss as tf_loss
from ray.rllib.agents.a3c.a3c_torch_policy import actor_critic_loss as torch_loss
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.tf_ops import explained_variance
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from MPE.cc_utils.mappo_tools import CentralizedValueMixin


tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


# Copied from A2C but optimizing the central value function.
def loss_with_central_critic_a2c(policy, model, dist_class, train_batch):
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


def central_vf_stats_a2c(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy._central_value_out)
    }
