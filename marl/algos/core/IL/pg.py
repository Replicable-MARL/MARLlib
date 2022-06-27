from ray.rllib.utils.framework import try_import_torch
from ray.rllib.agents.pg.pg_torch_policy import PGTorchPolicy
from ray.rllib.agents.pg.pg import DEFAULT_CONFIG as PG_CONFIG, PGTrainer

torch, nn = try_import_torch()

##########
### PG ###
##########


IPGTorchPolicy = PGTorchPolicy.with_updates(
    name="IPGTorchPolicy",
    get_default_config=lambda: PG_CONFIG,
)


def get_policy_class_ipg(config_):
    if config_["framework"] == "torch":
        return IPGTorchPolicy


IPGTrainer = PGTrainer.with_updates(
    name="IPGTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_ipg,
)
