from ray import tune

def run_ddpg(args, common_config, ma_config, cc_obs_dim, stop):

    results = tune.run(
        args.run,
        name=args.run + "_" + args.neural_arch + "_" + args.map,
        stop=stop,
        config=common_config,
        verbose=1
    )

    return results
