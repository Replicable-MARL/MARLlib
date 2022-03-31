import sys
from ray import tune
from ray.tune import register_env

from gym.spaces import Tuple

from pettingzoo.mpe import simple_adversary_v2, simple_crypto_v2, simple_v2, simple_push_v2, simple_tag_v2, \
    simple_spread_v2, simple_reference_v2, simple_world_comm_v2, simple_speaker_listener_v3
from MPE.env.mpe_rllib_qmix import RllibMPE_QMIX

def run_vdn_qmix(args, common_config, env_config, stop):

    if args.continues:
        print(
            "{} do not support continue action space".format(args.run)
        )
        sys.exit()

    if args.map not in ["simple_spread", "simple_speaker_listener", "simple_reference"]:
        print(
            "adversarial agents contained in this MPE scenario. "
            "Not suitable for cooperative only algo {}".format(args.run)
        )
        sys.exit()

    if args.neural_arch not in ["GRU"]:
        print("{} arch not supported for QMIX/VDN".format(args.neural_arch))
        sys.exit()

    if args.map == "simple_spread":
        env = simple_spread_v2.parallel_env(continuous_actions=False)
    elif args.map == "simple_reference":
        env = simple_reference_v2.parallel_env(continuous_actions=False)
    elif args.map == "simple_speaker_listener":
        env = simple_speaker_listener_v3.parallel_env(continuous_actions=False)
    else:
        print("not support QMIX/VDN in {}".format(args.map))
        sys.exit()

    test_env = RllibMPE_QMIX(env)
    agent_num = test_env.num_agents
    agent_list = test_env.agents
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    test_env.close()

    obs_space = Tuple([obs_space] * agent_num)
    act_space = Tuple([act_space] * agent_num)

    # align with RWARE/env/rware_rllib_qmix.py reset() function in line 41-50
    grouping = {
        "group_1": [i for i in agent_list],
    }

    # QMIX/VDN algo needs grouping env
    register_env(
        args.map,
        lambda _: RllibMPE_QMIX(env).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space))

    config = {
        "env": args.map,
        "train_batch_size": 32,
        "exploration_config": {
            "epsilon_timesteps": 5000,
            "final_epsilon": 0.05,
        },
        "mixer": "qmix" if args.run == "QMIX" else None,  # None for VDN, which has no mixer
        "num_gpus": args.num_gpus,
        "num_workers": args.num_workers,
        "num_gpus_per_worker": args.num_gpus_per_worker,

    }

    results = tune.run("QMIX",
                       name=args.run + "_" + args.neural_arch + "_" + args.map,
                       stop=stop,
                       config=config,
                       verbose=1)

    return results
