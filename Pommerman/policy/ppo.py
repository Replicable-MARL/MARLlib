from ray import tune


def run_ppo(args, common_config, env_config, agent_list, stop):

    config = {
        "env": "pommerman",
        "num_sgd_iter": 5,
    }

    config.update({
        "model": {
            "custom_model": args.neural_arch,
            "custom_model_config": {
                "agent_num": 4 if "One" not in args.map else 2,
                "map_size": 11 if "One" not in args.map else 8,
            }
        },
    })

    config.update(common_config)
    results = tune.run(args.run,
                       name=args.run + "_" + args.neural_arch + "_" + args.map,
                       stop=stop,
                       config=config,
                       verbose=1)

    return results
