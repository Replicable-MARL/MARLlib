from gym.spaces import Dict as GymDict, Tuple, Box, Discrete
from ray import tune
from ray.tune.registry import register_env
from Pommerman.env.pommerman_rllib_qmix import RllibPommerman_QMIX
from Pommerman.model.torch_qmix_model import QMixTrainer


def run_vdn_qmix(args, common_config, env_config, agent_list, stop):
    if args.neural_arch not in ["CNN_GRU", ]:
        print("{} arch not supported for QMIX/VDN".format(args.neural_arch))
        raise ValueError()

    if "Team" not in args.map:
        print("QMIX/VDN is only for cooperative scenarios")
        raise ValueError()

    if env_config["neural_agent_pos"] == [0, 1, 2, 3]:
        # 2 vs 2
        grouping = {
            "group_1": ["agent_{}".format(i) for i in [0, 1]],
            "group_2": ["agent_{}".format(i) for i in [2, 3]],
        }

    elif env_config["neural_agent_pos"] in [[0, 1], [2, 3]]:
        grouping = {
            "group_1": ["agent_{}".format(i) for i in [0, 1]],
        }

    else:
        print("Wrong agent position setting")
        raise ValueError

    # simulate one single env
    single_env = RllibPommerman_QMIX(env_config, agent_list)
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    obs_space = Tuple([obs_space] * 2)
    act_space = Tuple([act_space] * 2)

    # QMIX/VDN algo needs grouping env
    register_env(
        "grouped_pommerman",
        lambda _: RllibPommerman_QMIX(env_config, agent_list).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space))

    config = {
        "env": "grouped_pommerman",
        "train_batch_size": 32,
        "exploration_config": {
            "epsilon_timesteps": 5000,
            "final_epsilon": 0.05,
        },
        "model": {
            "custom_model_config": {
                "neural_arch": args.neural_arch,
                "map_size": 11 if "One" not in args.map else 8,
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
