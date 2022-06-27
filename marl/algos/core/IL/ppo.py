from ray.rllib.utils.framework import try_import_torch
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer as PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG

torch, nn = try_import_torch()

###########
### PPO ###
###########


IPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="IPPOTorchPolicy",
    get_default_config=lambda: PPO_CONFIG,
)


def get_policy_class_ppo(config_):
    if config_["framework"] == "torch":
        return IPPOTorchPolicy


IPPOTrainer = PPOTrainer.with_updates(
    name="IPPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_ppo,
)
