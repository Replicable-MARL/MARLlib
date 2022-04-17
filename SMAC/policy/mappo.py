from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.tune.utils import merge_dicts
from SMAC.util.mappo_tools import *
from SMAC.util.maa2c_tools import *


def run_mappo(args, common_config, env_config, stop, reporter):
    """
           for bug mentioned https://github.com/ray-project/ray/pull/20743
           make sure sgd_minibatch_size > max_seq_len
           """

    obs_shape = env_config["obs_shape"]
    n_ally = env_config["n_ally"]
    n_enemy = env_config["n_enemy"]
    state_shape = env_config["state_shape"]
    n_actions = env_config["n_actions"]
    episode_limit = env_config["episode_limit"]
    episode_num = 10
    iteration = 4
    train_batch_size = episode_num * episode_limit // args.batchsize_reduce

    sgd_minibatch_size = train_batch_size
    while sgd_minibatch_size < episode_limit:
        sgd_minibatch_size *= 2

    config = {
        "env": "smac",
        "train_batch_size": train_batch_size,
        "num_sgd_iter": iteration,
        "sgd_minibatch_size": sgd_minibatch_size,
        "batch_mode": "complete_episodes",
        "entropy_coeff": 0.01,
        "clip_param": 0.2,
        "vf_clip_param": 20.0,  # very sensitive, depends on the scale of the rewards
        "lr": 0.0005,
        "model": {
            "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
            "max_seq_len": episode_limit,
            "custom_model_config": {
                "token_dim": args.token_dim,
                "ally_num": n_ally,
                "enemy_num": n_enemy,
                "self_obs_dim": obs_shape,
                "state_dim": state_shape
            },
        },
    }

    config.update(common_config)

    MAPPO_CONFIG = merge_dicts(
        PPO_CONFIG,
        {
            "agent_num": n_ally,
            "state_dim": state_shape,
            "self_obs_dim": obs_shape,
            "centralized_critic_obs_dim": -1,
        }
    )

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
        get_default_config=lambda: MAPPO_CONFIG,
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

    results = tune.run(MAPPOTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                       config=config,
                       verbose=1,
                       progress_reporter=reporter
                       )

    return results
