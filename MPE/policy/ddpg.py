import sys
import ray
from ray import tune



def run_ddpg(args, common_config, env_config, stop):

    if not args.continues:
        print(
            "{} only support continues action space".format(args.run)
        )
        sys.exit()

    results = tune.run(
        args.run,
        name=args.run + "_" + args.neural_arch + "_" + args.map,
        stop=stop,
        config=common_config,
        verbose=1
    )

    return results