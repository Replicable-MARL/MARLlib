
from ray import tune

from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.a3c.a3c import DEFAULT_CONFIG as A3C_CONFIG
from MPE.util.mappo_tools import *
from MPE.util.maa2c_tools import *



def run_maa2c(args, common_config, env_config, stop):
    config = {
        "model": {
            "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
            "custom_model_config": {
                "agent_num": env_config["n_agents"]
            },
        },
    }
    config.update(common_config)

    MAA2CTFPolicy = A3CTFPolicy.with_updates(
        name="MAA2CTFPolicy",
        postprocess_fn=centralized_critic_postprocessing,
        loss_fn=loss_with_central_critic_a2c,
        grad_stats_fn=central_vf_stats_a2c,
        mixins=[
            CentralizedValueMixin
        ])

    MAA2CTorchPolicy = A3CTorchPolicy.with_updates(
        name="MAA2CTorchPolicy",
        get_default_config=lambda: A3C_CONFIG,
        postprocess_fn=centralized_critic_postprocessing,
        loss_fn=loss_with_central_critic_a2c,
        mixins=[
            CentralizedValueMixin
        ])

    def get_policy_class(config_):
        if config_["framework"] == "torch":
            return MAA2CTorchPolicy

    MAA2CTrainer = A2CTrainer.with_updates(
        name="MAA2CTrainer",
        default_policy=MAA2CTFPolicy,
        get_policy_class=get_policy_class,
    )

    results = tune.run(MAA2CTrainer,
                       name=args.run + "_" + args.neural_arch + "_" + args.map,
                       stop=stop,
                       config=config,
                       verbose=1)

    return results
