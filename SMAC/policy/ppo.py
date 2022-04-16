from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG


def run_ppo(args, common_config, env_config, stop, reporter):
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
    train_batch_size = episode_num * episode_limit
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
            "custom_model": "{}_IndependentCritic".format(args.neural_arch),
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

    PPOTrainer_ = PPOTrainer.with_updates(
        default_config=PPO_CONFIG,
    )

    results = tune.run(PPOTrainer_, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop, config=config,
                       verbose=1, progress_reporter=reporter)

    return results
