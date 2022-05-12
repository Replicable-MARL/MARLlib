from ray import tune

from ray.rllib.agents.dqn.r2d2 import DEFAULT_CONFIG, R2D2Trainer
from ray.rllib.agents.dqn.r2d2_tf_policy import R2D2TFPolicy

from Hanabi.model.torch_mask_r2d2 import *


def run_r2d2(args, common_config, n_agents, stop):
    config = {
        "model": {
            "custom_model": "{}_IndependentCritic".format(args.neural_arch),
        },
        "framework": args.framework,
    }
    config.update(common_config)

    def get_policy_class(config_):
        if config_["framework"] == "torch":
            return R2D2WithMaskPolicy

    DEFAULT_CONFIG['dueling'] = False  # with mask, only support no dueling arch, default use dueling arch

    R2D2Trainer_ = R2D2Trainer.with_updates(
        name="R2D2_Trainer",
        default_config=DEFAULT_CONFIG,
        default_policy=R2D2TFPolicy,
        get_policy_class=get_policy_class,
    )

    results = tune.run(R2D2Trainer_, name=args.run + "_" + args.neural_arch + "_" + "Hanabi", stop=stop,
                       config=config,
                       verbose=1)

    return results
