from ray import tune
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.a3c.a2c import A2CTrainer

from Pommerman.util.mappo_tools import *
from Pommerman.util.maa2c_tools import *


def run_coma(args, common_config, env_config, agent_list, stop):
    if env_config["rule_agent_pos"] != []:
        print("Centralized critic can not be used in scenario where rule"
              "based agent exists"
              "\n Set rule_agent_pos to empty list []")
        raise ValueError()

    if "Team" not in args.map:
        print("Scenario \"{}\" is under fully observed setting. "
              "MAA2C is not suitable".format(args.map))
        raise ValueError()

    config = {
        "env": "pommerman",
        "model": {
            "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
            "custom_model_config": {
                "agent_num": 4 if "One" not in args.map else 2,
                "map_size": 11 if "One" not in args.map else 8,
                "coma": True
            },
        },
    }
    config.update(common_config)

    from Pommerman.util.coma_tools import loss_with_central_critic_coma, central_vf_stats_coma, COMATorchPolicy

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
