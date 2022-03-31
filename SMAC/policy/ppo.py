from ray import tune


def run_ppo(args, common_config, env_config, stop):
    """
            for bug mentioned https://github.com/ray-project/ray/pull/20743
            make sure sgd_minibatch_size > max_seq_len
            """
    obs_shape = env_config["obs_shape"]
    n_ally = env_config["n_ally"]
    n_enemy = env_config["n_enemy"]
    state_shape = env_config["state_shape"]
    n_actions = env_config["n_actions"]
    rollout_fragment_length = env_config["rollout_fragment_length"]

    sgd_minibatch_size = 128
    while sgd_minibatch_size < rollout_fragment_length:
        sgd_minibatch_size *= 2

    config = {
        "env": "smac",
        "num_sgd_iter": args.num_sgd_iter,
        "sgd_minibatch_size": sgd_minibatch_size,
        "model": {
            "custom_model": "{}_IndependentCritic".format(args.neural_arch),
            "max_seq_len": rollout_fragment_length,
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
    results = tune.run(args.run, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop, config=config,
                       verbose=1)

    return results
