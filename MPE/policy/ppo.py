from ray import tune


def run_ppo(args, common_config, env_config, stop):
    config = {
        "model": {
            "custom_model": "{}_IndependentCritic".format(args.neural_arch),
        },
        "num_sgd_iter": 10,
    }

    config.update(common_config)

    results = tune.run(
        args.run,
        name=args.run + "_" + args.neural_arch + "_" + args.map,
        stop=stop,
        config=config,
        verbose=1
    )

    return results
