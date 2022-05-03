
import sys
from ray import tune
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.a3c.a2c import A2CTrainer
from MPE.util.mappo_tools import *
from MPE.util.maa2c_tools import *


def run_coma(args, common_config, env_config, stop):
    if args.continues:
        print("continues action space not supported in COMA")
        sys.exit()

    config = {
        "model": {
            "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
            "custom_model_config": {
                "agent_num": env_config["n_agents"],
                "coma": True
            },
        },
    }
    config.update(common_config)

    from MPE.util.coma_tools import loss_with_central_critic_coma, central_vf_stats_coma, COMATorchPolicy

    # not used
    COMATFPolicy = A3CTFPolicy.with_updates(
        name="MAA2CTFPolicy",
        postprocess_fn=centralized_critic_postprocessing,
        loss_fn=loss_with_central_critic_coma,
        grad_stats_fn=central_vf_stats_coma,
        mixins=[
            CentralizedValueMixin
        ])

    COMATorchPolicy = COMATorchPolicy.with_updates(
        name="MAA2CTorchPolicy",
        loss_fn=loss_with_central_critic_coma,
        mixins=[
            CentralizedValueMixin
        ])

    def get_policy_class(config_):
        if config_["framework"] == "torch":
            return COMATorchPolicy

    COMATrainer = A2CTrainer.with_updates(
        name="COMATrainer",
        default_policy=COMATFPolicy,
        get_policy_class=get_policy_class,
    )

    results = tune.run(COMATrainer,
                       name=args.run + "_" + args.neural_arch + "_" + args.map,
                       stop=stop,
                       config=config,
                       verbose=1)

    return results
