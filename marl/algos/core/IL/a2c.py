from ray.rllib.utils.framework import try_import_torch
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG, A2CTrainer

torch, nn = try_import_torch()

###########
### A2C ###
###########


IA2CTorchPolicy = A3CTorchPolicy.with_updates(
    name="IA2CTorchPolicy",
    get_default_config=lambda: A2C_CONFIG,
)


def get_policy_class_ia2c(config_):
    if config_["framework"] == "torch":
        return IA2CTorchPolicy


IA2CTrainer = A2CTrainer.with_updates(
    name="IA2CTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_ia2c,
)
