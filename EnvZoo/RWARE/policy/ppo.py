from ray import tune


def run_ppo(args, common_config, env_config, map_name, stop):

    config = {"num_sgd_iter": 5, }

    if "_" in args.neural_arch:
        config.update({
            "model": {
                "custom_model": "{}_IndependentCritic".format(args.neural_arch),
            },
        })

    config.update(common_config)
    results = tune.run(args.run,
                       name=args.run + "_" + args.neural_arch + "_" + map_name,
                       stop=stop,
                       config=config,
                       verbose=1)

    return results
