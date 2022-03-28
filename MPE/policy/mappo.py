from ray import tune
from ray.tune import register_env
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from MPE.util.mappo_tools import *
from MPE.util.maa2c_tools import *
from MPE.util.vdppo_tools import *




def run_mappo(args, common_config, env_config, stop):
    config = {
        "model": {
            "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
            "custom_model_config": {
                "agent_num": env_config["n_agents"]
            }
        },
        "num_sgd_iter": 10,
    }
    config.update(common_config)

    MAPPOTFPolicy = PPOTFPolicy.with_updates(
        name="MAPPOTFPolicy",
        postprocess_fn=centralized_critic_postprocessing,
        loss_fn=loss_with_central_critic,
        before_loss_init=setup_tf_mixins,
        grad_stats_fn=central_vf_stats_ppo,
        mixins=[
            LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
            CentralizedValueMixin
        ])

    MAPPOTorchPolicy = PPOTorchPolicy.with_updates(
        name="MAPPOTorchPolicy",
        get_default_config=lambda: PPO_CONFIG,
        postprocess_fn=centralized_critic_postprocessing,
        loss_fn=loss_with_central_critic,
        before_init=setup_torch_mixins,
        mixins=[
            TorchLR, TorchEntropyCoeffSchedule, TorchKLCoeffMixin,
            CentralizedValueMixin
        ])

    def get_policy_class(config_):
        if config_["framework"] == "torch":
            return MAPPOTorchPolicy

    MAPPOTrainer = PPOTrainer.with_updates(
        name="MAPPOTrainer",
        default_policy=MAPPOTFPolicy,
        get_policy_class=get_policy_class,
    )

    results = tune.run(MAPPOTrainer,
                       name=args.run + "_" + args.neural_arch + "_" + args.map,
                       stop=stop,
                       config=config,
                       verbose=1)

    return results
