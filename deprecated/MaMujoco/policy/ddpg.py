from ray import tune

def run_ddpg(args, common_config, env_config, stop):

    config = {
        "env": args.map,
        "horizon": args.horizon,
    }
    config.update(common_config)

    results = tune.run(
        args.run,
        name=args.run + "_" + "MLP" + "_" + args.map,
        stop=stop,
        config=config,
        verbose=1
    )

    return results
