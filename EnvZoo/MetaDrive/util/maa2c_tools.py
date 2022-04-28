from MetaDrive.util.mappo_tools import *
from ray.rllib.agents.a3c.a3c_tf_policy import actor_critic_loss as tf_loss
from ray.rllib.agents.a3c.a3c_torch_policy import actor_critic_loss as torch_loss

# Copied from A2C but optimizing the central value function.
def loss_with_central_critic_a2c(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = tf_loss if not policy.config["framework"] == "torch" else torch_loss

    vf_saved = model.value_function
    model.value_function = lambda: policy.model.central_value_function(train_batch["centralized_critic_obs"])
    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)
    model.value_function = vf_saved
    return loss


def central_vf_stats_a2c(policy, train_batch):
    # Report the explained variance of the central value function.
    return {
        "value_targets": torch.mean(train_batch[Postprocessing.VALUE_TARGETS]),
        "advantage_mean": torch.mean(train_batch[Postprocessing.ADVANTAGES]),
        "advantages_min": torch.min(train_batch[Postprocessing.ADVANTAGES]),
        "advantages_max": torch.max(train_batch[Postprocessing.ADVANTAGES]),
        "central_value_mean": torch.mean(policy._central_value_out),
        "central_value_min": torch.min(policy._central_value_out),
        "central_value_max": torch.max(policy._central_value_out),
        "policy_entropy": torch.mean(
            torch.stack(policy.get_tower_stats("entropy"))),
        "policy_loss": torch.mean(
            torch.stack(policy.get_tower_stats("pi_err"))),
        "vf_loss": torch.mean(
            torch.stack(policy.get_tower_stats("value_err"))),
    }

