from ray import tune


def run_ppo(args, common_config, n_agents, stop):
    config = {
        "model": {
            "custom_model": "{}_IndependentCritic".format(args.neural_arch),
        },
        "num_sgd_iter": 5,
    }

    config.update(common_config)

    results = tune.run(
        args.run,
        name=args.run + "_" + args.neural_arch + "_" + "Hanabi",
        stop=stop,
        config=config,
        verbose=1
    )

    return results
