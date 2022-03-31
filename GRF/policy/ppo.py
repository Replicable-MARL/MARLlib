from ray import tune


def run_ppo(args, common_config, env_config, stop):
    config = {
        "env": "football",
        "num_sgd_iter": 5,
    }

    if "_" in args.neural_arch:
        config.update({
            "model": {
                "custom_model": args.neural_arch,
            },
        })

    config.update(common_config)
    results = tune.run(args.run,
                       name=args.run + "_" + args.neural_arch + "_" + args.map,
                       stop=stop,
                       config=config,
                       verbose=1)

    return results
