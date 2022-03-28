from ray.rllib.models import ModelCatalog
import os
import sys
import ray
from ray import tune
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer
from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy, ComputeTDErrorMixin
from ray.rllib.agents.ddpg.ddpg_tf_policy import DDPGTFPolicy
from ray.rllib.agents.ddpg.ddpg import DEFAULT_CONFIG as DDPG_CONFIG
from MPE.util.maddpg_tools import maddpg_actor_critic_loss, build_maddpg_models_and_action_dist, \
    maddpg_centralized_critic_postprocessing
from ray.rllib.agents.sac.sac_torch_policy import TargetNetworkMixin
from MPE.util.maa2c_tools import *




def run_maddpg(args, common_config, env_config, stop):
    if not args.continues:
        print(
            "{} only support continues action space".format(args.run)
        )
        sys.exit()

    from MPE.model.torch_maddpg import MADDPGTorchModel
    ModelCatalog.register_custom_model(
        "torch_maddpg", MADDPGTorchModel)

    config = {
        "model": {
            "custom_model": "torch_maddpg",
            "custom_model_config": {
                "agent_num": env_config["n_agents"]
            },
        },
    }
    config.update(common_config)

    MADDPGTFPolicy = DDPGTFPolicy.with_updates(
        name="MADDPGTFPolicy",
        postprocess_fn=maddpg_centralized_critic_postprocessing,
        loss_fn=maddpg_actor_critic_loss,
        mixins=[
            TargetNetworkMixin,
            ComputeTDErrorMixin,
            CentralizedValueMixin
        ])

    MADDPGTorchPolicy = DDPGTorchPolicy.with_updates(
        name="MADDPGTorchPolicy",
        get_default_config=lambda: DDPG_CONFIG,
        postprocess_fn=maddpg_centralized_critic_postprocessing,
        make_model_and_action_dist=build_maddpg_models_and_action_dist,
        loss_fn=maddpg_actor_critic_loss,
        mixins=[
            TargetNetworkMixin,
            ComputeTDErrorMixin,
            CentralizedValueMixin
        ])

    def get_policy_class(config_):
        if config_["framework"] == "torch":
            return MADDPGTorchPolicy

    MADDPGTrainer = DDPGTrainer.with_updates(
        name="MADDPGTrainer",
        default_policy=MADDPGTFPolicy,
        get_policy_class=get_policy_class,
    )

    results = tune.run(MADDPGTrainer,
                       name=args.run + "_" + args.map,
                       stop=stop,
                       config=config,
                       verbose=1)

    return results