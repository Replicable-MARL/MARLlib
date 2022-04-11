from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.utils.torch_ops import apply_grad_clipping

from MaMujoco.util.mappo_tools import *
from MaMujoco.util.maa2c_tools import *
from MaMujoco.util.vdppo_tools import *


def run_mappo(args, common_config, env_config, stop):
    """
            for bug mentioned https://github.com/ray-project/ray/pull/20743
            make sure sgd_minibatch_size > max_seq_len
            """
    sgd_minibatch_size = 128
    while sgd_minibatch_size < args.horizon:
        sgd_minibatch_size *= 2

    # config = {
    #     "env": args.map,
    #     "horizon": args.horizon,
    #     "num_sgd_iter": 5,
    #     "sgd_minibatch_size": sgd_minibatch_size,
    #     "model": {
    #         "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
    #         "custom_model_config": {
    #             "agent_num": env_config["ally_num"],
    #             "state_dim": env_config["state_dim"]
    #         }
    #     },
    # }

    config = dict()
    config.update(common_config)

    config.update({
        "seed": 1,
        "env": args.map,
        "horizon": 1000,
        "num_sgd_iter": 5,  # ppo-epoch
        "train_batch_size": 4000,
        "sgd_minibatch_size": sgd_minibatch_size,
        "lr": 5e-5,
        "grad_clip": 20,
        "clip_param": 0.3,
        "model": {
            "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
            "custom_model_config": {
                "agent_num": env_config["ally_num"],
                "state_dim": env_config["state_dim"]
            },
            "vf_share_layers": True,
        },
    })

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
        extra_grad_process_fn=apply_grad_clipping,
        mixins=[
            TorchLR, TorchEntropyCoeffSchedule, TorchKLCoeffMixin,
            CentralizedValueMixin
        ])

    def get_policy_class(config_):
        if config_["framework"] == "torch":
            return MAPPOTorchPolicy

    MAPPOTrainer = PPOTrainer.with_updates(
        name="#lr-5e-5-config-as-with-Grad-Norm-Clip-20-With-Fn-MAPPOTrainer#",
        default_policy=MAPPOTFPolicy,
        get_policy_class=get_policy_class,
    )

    results = tune.run(MAPPOTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                       config=config,
                       verbose=1)

    return results
