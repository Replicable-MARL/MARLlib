from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

from MetaDrive.util.mappo_tools import *
from MetaDrive.util.maa2c_tools import *


def run_mappo(args, common_config, ma_config, cc_obs_dim, stop):
    """
            for bug mentioned https://github.com/ray-project/ray/pull/20743
            make sure sgd_minibatch_size > max_seq_len
            """
    config = {
        "model": {
            "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
        },
        "num_sgd_iter": 10,
    }
    config.update(common_config)

    def get_policy_class(config):
        if config["framework"] == "torch":
            return MAPPOTorchPolicy
        else:
            raise ValueError()

    ma_config.update(PPO_CONFIG)

    ma_config["centralized_critic_obs_dim"] = cc_obs_dim

    MAPPOTorchPolicy = PPOTorchPolicy.with_updates(
        name="MAPPOTorchPolicy",
        get_default_config=lambda: ma_config,
        make_model=make_model,
        extra_action_out_fn=vf_preds_fetches,
        postprocess_fn=centralized_critic_postprocessing,
        loss_fn=loss_with_central_critic_ppo,
        stats_fn=central_vf_stats_ppo,
        before_init=setup_torch_mixins,
        mixins=[TorchLR, TorchEntropyCoeffSchedule, TorchKLCoeffMixin, CentralizedValueMixin]
    )

    MAPPOTrainer = PPOTrainer.with_updates(
        name="MAPPOTrainer",
        default_config=ma_config,
        default_policy=MAPPOTorchPolicy,
        get_policy_class=get_policy_class,
    )

    results = tune.run(MAPPOTrainer,
                       name=args.run + "_" + args.neural_arch + "_" + args.map,
                       stop=stop,
                       config=config,
                       verbose=1)

    return results
