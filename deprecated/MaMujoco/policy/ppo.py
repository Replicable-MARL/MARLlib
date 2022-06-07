from ray import tune


def run_ppo(args, common_config, env_config, stop):
    """
            for bug mentioned https://github.com/ray-project/ray/pull/20743
            make sure sgd_minibatch_size > max_seq_len
            """
    sgd_minibatch_size = 128
    while sgd_minibatch_size < args.horizon:
        sgd_minibatch_size *= 2

    config = {
        "env": args.map,
        "horizon": args.horizon,
        "num_sgd_iter": 5,
        "sgd_minibatch_size": sgd_minibatch_size,
        "model": {
            "custom_model": args.neural_arch,
        },
    }
    config.update(common_config)
    results = tune.run(args.run, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop, config=config,
                       verbose=1)

    return results
