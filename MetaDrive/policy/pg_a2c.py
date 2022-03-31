from ray import tune


def run_pg_a2c(args, common_config, ma_config, cc_obs_dim, stop):
    config = {
        "model": {
            "custom_model": "{}_IndependentCritic".format(args.neural_arch),
        },
    }

    config.update(common_config)

    results = tune.run(args.run,
             name=args.run + "_" + args.neural_arch + "_" + args.map,
             stop=stop,
             config=config,
             verbose=1)

    return results
