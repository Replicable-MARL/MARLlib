import argparse


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-mode",
        # default=True,
        default=False,
        type=bool,
        help="Init Ray in local mode for easier debugging.")
    parser.add_argument(
        "--parallel",
        # default=False,
        default=False,
        type=bool,
        help="Whether use tune grid research")
    parser.add_argument(
        "--run",
        choices=["PG", "A2C", "A3C", "MAA2C", "DDPG", "MADDPG", "PPO", "MAPPO", "SUM-VDA2C", "MIX-VDA2C", "SUM-VDPPO", "MIX-VDPPO", "HAPPO"],  # "APPO" "IMPALA"
        # choices=["HAPPO"],  # "APPO" "IMPALA"
        # default="A2C",
        default="HAPPO",
        # default="MAPPO",
        help="The RLlib-registered algorithm to use.")
    parser.add_argument(
        "--share-policy",
        type=bool,
        default=False,
        # default=True,
        help="Maps should be registered")
    parser.add_argument(
        "--map",
        choices=[
            "2AgentAnt",
            "2AgentAntDiag",
            "4AgentAnt",
            "2AgentHalfCheetah",
            "6AgentHalfCheetah",
            "3AgentHopper",
            "2AgentHumanoid",
            "2AgentHumanoidStandup",
            "2AgentReacher",
            "2AgentSwimmer",
            "2AgentWalker",
            "ManyagentSwimmer",
            "ManyagentAnt",
        ],
        # default="ManyagentAnt",
        default="2AgentHalfCheetah",
        # default="2AgentWalker",
        help="Envs should be registered")
    parser.add_argument(
        "--neural-arch",
        choices=["LSTM", "GRU"],
        type=str,
        default="GRU",
        help="Agent Neural Architecture")
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "tfe", "torch"],
        default="torch",
        help="The DL framework specifier. Use torch please")
    parser.add_argument(
        "--horizon",
        type=int,
        default=500,
        help="episode limit, terminate after this step")
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=0,
        help="GPU number per trail. 0 for max")
    parser.add_argument(
        "--num-gpus",
        type=float,
        default=0.2,
        help="GPU number per trail")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Sampler number per trail")
    parser.add_argument(
        "--num-cpus-per-worker",
        type=float,
        default=1)
    parser.add_argument(
        "--num-gpus-per-worker",
        type=float,
        default=0.1)
    parser.add_argument(
        "--num-gpus-per-trial",
        type=float,
        default=1)
    parser.add_argument(
        "--stop-iters",
        type=int,
        default=100000,
        help="Number of iterations to train.")
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=int(1e7),
        help="Number of timesteps to train.")
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=99999,
        help="Reward at which we stop training.")
    parser.add_argument(
        "--test",
        action="store_true")
    return parser
