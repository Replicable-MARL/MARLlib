import argparse


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-mode",
        default=True,
        type=bool,
        help="Init Ray in local mode for easier debugging.")
    parser.add_argument(
        "--parallel",
        default=False,
        type=bool,
        help="Whether use tune grid research")
    parser.add_argument(
        "--run",
        choices=["QMIX", "VDN", "R2D2", "PG", "A2C", "A3C", "MAA2C", "PPO", "MAPPO", "COMA", "SUM-VDA2C", "MIX-VDA2C", "SUM-VDPPO", "MIX-VDPPO"],  # "APPO" "IMPALA"
        default="MIX-VDPPO",
        help="The RLlib-registered algorithm to use.")
    parser.add_argument(
        "--agents-num",
        type=int,
        default=4,
        help="range(1, 20)")
    parser.add_argument(
        "--map-size",
        choices=["tiny", "small", "medium", "large"],
        default="small",
        help="")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="easy",
        help="")
    parser.add_argument(
        "--neural-arch",
        choices=["LSTM", "GRU"],
        type=str,
        default="LSTM",
        help="Agent Neural Architecture")
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "tfe", "torch"],
        default="torch",
        help="The DL framework specifier. Use torch please")
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
        default=2,
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
        default=1000000,
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
