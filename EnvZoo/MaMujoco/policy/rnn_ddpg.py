from ray import tune
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy

from MaMujoco.util.ddpg_tools import *


def run_rnnddpg(args, common_config, env_config, stop):

    RNNDDPGTorchPolicy = DDPGTorchPolicy.with_updates(
        name="RNNDDPGTorchPolicy",
        get_default_config=lambda: RNNDDPG_DEFAULT_CONFIG,
        action_distribution_fn=action_distribution_fn,
        make_model_and_action_dist=build_rnnddpg_models_and_action_dist,
        loss_fn=ddpg_actor_critic_loss,
    )

    def get_policy_class(config: TrainerConfigDict) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            return RNNDDPGTorchPolicy

    RNNDDPGTrainer = DDPGTrainer.with_updates(
        name="RNNDDPGTrainer",
        default_config=RNNDDPG_DEFAULT_CONFIG,
        default_policy=RNNDDPGTorchPolicy,
        get_policy_class=get_policy_class,
        validate_config=validate_config,
        allow_unknown_subkeys=["Q_model", "policy_model"]
    )

    config = {
        "env": args.map,
        "horizon": args.horizon,
    }
    config.update(common_config)

    config["env"] = args.map
    config["horizon"] = args.horizon
    print(config)

    results = tune.run(
        RNNDDPGTrainer,
        name=args.run + "_" + "MLP" + "_" + args.map,
        stop=stop,
        config=config,
        verbose=1
    )

    return results
