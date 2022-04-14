from ray import tune
from ray.rllib.agents.trainer import with_common_config


def run_pg_a2c_a3c(args, common_config, env_config, stop, reporter):
    obs_shape = env_config["obs_shape"]
    n_ally = env_config["n_ally"]
    n_enemy = env_config["n_enemy"]
    state_shape = env_config["state_shape"]
    n_actions = env_config["n_actions"]
    episode_limit = env_config["episode_limit"]

    episode_num = 10
    train_batch_size = episode_num * episode_limit
    config = {
        "env": "smac",
        "batch_mode": "complete_episodes",
        "train_batch_size": train_batch_size,
        "lr": 0.0005,
        "model": {
            "custom_model": "{}_IndependentCritic".format(args.neural_arch),
            "max_seq_len": episode_limit + 1,
            "custom_model_config": {
                "token_dim": args.token_dim,
                "ally_num": n_ally,
                "enemy_num": n_enemy,
                "self_obs_dim": obs_shape,
                "state_dim": state_shape
            },
        },
    }

    if args.run != "PG":
        config["entropy_coeff"] = 0.01

    config.update(common_config)

    results = tune.run(args.run, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop, config=config,
                       verbose=1, progress_reporter=reporter)

    return results
