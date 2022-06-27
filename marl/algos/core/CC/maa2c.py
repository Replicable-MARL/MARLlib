from ray.rllib.utils.framework import try_import_torch
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy, actor_critic_loss
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG, A2CTrainer
from marl.algos.utils.centralized_critic import CentralizedValueMixin, centralized_critic_postprocessing

torch, nn = try_import_torch()


#############
### MAA2C ###
#############

# Copied from A2C but optimizing the central value function.
def central_critic_a2c_loss(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = actor_critic_loss

    vf_saved = model.value_function
    opp_action_in_cc = policy.config["model"]["custom_model_config"]["opp_action_in_cc"]
    model.value_function = lambda: policy.model.central_value_function(train_batch["state"],
                                                                       train_batch[
                                                                           "opponent_actions"] if opp_action_in_cc else None)

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


