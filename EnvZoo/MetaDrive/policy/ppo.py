from ray import tune


def run_ppo(args, common_config, ma_config, cc_obs_dim, stop):
    """
            for bug mentioned https://github.com/ray-project/ray/pull/20743
            make sure sgd_minibatch_size > max_seq_len
            """

    config = {
        "model": {
            "custom_model": "{}_IndependentCritic".format(args.neural_arch),
        },
        "num_sgd_iter": 10,
    }

    config.update(common_config)

    results = tune.run(args.run,
                       name=args.run + "_" + args.neural_arch + "_" + args.map,
                       stop=stop,
                       config=config,
                       verbose=1)

    return results
