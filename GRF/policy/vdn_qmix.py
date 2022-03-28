from gym.spaces import Dict as GymDict, Tuple, Box, Discrete

from ray import tune
from ray.tune.registry import register_env

from GRF.env.football_rllib_qmix import RllibGFootball_QMIX

from GRF.model.torch_qmix_model import QMixTrainer


def run_vdn_qmix(args, common_config, env_config, stop):

    if args.neural_arch not in ["CNN_GRU", "CNN_UPDeT"]:
        print("{} arch not supported for QMIX/VDN".format(args.neural_arch))
        raise ValueError()

    single_env = RllibGFootball_QMIX(env_config)
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    obs_space = Tuple([obs_space] * env_config["num_agents"])
    act_space = Tuple([act_space] * env_config["num_agents"])

    # align with GoogleFootball/env/football_rllib_qmix.py reset() function in line 41-50
    grouping = {
        "group_1": ["agent_{}".format(i) for i in range(env_config["num_agents"])],
    }

    # QMIX/VDN algo needs grouping env
    register_env(
        "grouped_football",
        lambda _: RllibGFootball_QMIX(env_config).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space))

    config = {
        "env": "grouped_football",
        "train_batch_size": 32,
        "exploration_config": {
            "epsilon_timesteps": 5000,
            "final_epsilon": 0.05,
        },
        "model": {
            "custom_model_config": {
                "neural_arch": args.neural_arch,
            },
        },
        "mixer": "qmix" if args.run == "QMIX" else None,  # None for VDN, which has no mixer
        "num_gpus": args.num_gpus,
        "num_workers": args.num_workers,
        "num_gpus_per_worker": args.num_gpus_per_worker,

    }

    results = tune.run(QMixTrainer,
                       name=args.run + "_" + args.neural_arch + "_" + args.map,
                       stop=stop,
                       config=config,
                       verbose=1)

    return results