from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2CTrainer

from MetaDrive.util.maa2c_tools import *


def run_maa2c(args, common_config, ma_config, cc_obs_dim, stop):
    config = {
        "model": {
            "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
        }
    }

    config.update(common_config)

    def get_policy_class(config):
        if config["framework"] == "torch":
            return MAA2CTorchPolicy
        else:
            raise ValueError()

    ma_config.update(A2C_CONFIG)

    ma_config["centralized_critic_obs_dim"] = cc_obs_dim

    MAA2CTorchPolicy = A3CTorchPolicy.with_updates(
        name="MAA2CTorchPolicy",
        get_default_config=lambda: ma_config,
        make_model=make_model,
        extra_action_out_fn=vf_preds_fetches,
        postprocess_fn=centralized_critic_postprocessing,
        loss_fn=loss_with_central_critic_a2c,
        stats_fn=central_vf_stats_a2c,
        mixins=[CentralizedValueMixin]
    )

    MAA2CTrainer = A2CTrainer.with_updates(
        name="MAA2CTrainer",
        default_config=ma_config,
        default_policy=MAA2CTorchPolicy,
        get_policy_class=get_policy_class,
    )

    results = tune.run(MAA2CTrainer,
                       name=args.run + "_" + args.neural_arch + "_" + args.map,
                       stop=stop,
                       config=config,
                       verbose=1)

    return results
