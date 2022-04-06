import argparse


def get_train_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--local-mode",
    #     default=True,
    #     type=bool,
    #     help="Init Ray in local mode for easier debugging.")
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.")
    parser.add_argument(
        "--run",
        choices=["IQL", "QMIX", "VDN", "R2D2", "PG", "A2C", "A3C", "MAA2C", "PPO", "MAPPO", "COMA", "MIX-VDA2C", "SUM-VDA2C", "MIX-VDPPO", "SUM-VDPPO"],
        # "APPO" "IMPALA"
        default="VDN",
        help="The RLlib-registered algorithm to use.")
    parser.add_argument(
        "--map",
        type=str,
        default="3m",
        help="Maps should be registered")
    # parser.add_argument(
    #     "--share-policy",
    #     type=bool,
    #     default=True,
    #     help="Maps should be registered")
    parser.add_argument(
        "--share-policy",
        action="store_true",
        help="Maps should be registered")
    parser.add_argument(
        "--token-dim",
        type=int,
        default=5,
        help="Maps should be registered")
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        default=10,
        help="evaluation_interval")
    parser.add_argument(
        "--neural-arch",
        choices=["LSTM", "GRU", "UPDeT", ],
        type=str,
        default="GRU",
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
        default=0.5,
        help="GPU number per trail")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
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
        "--num-seeds",
        type=int,
        default=3)
    parser.add_argument(
        "--stop-iters",
        type=int,
        default=200000,
        help="Number of iterations to train.")
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=2000000,
        help="Number of timesteps to train.")
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=99999,
        help="Reward at which we stop training.")
    parser.add_argument(
        "--exp-name-prefix",
        type=str,
        default="")
    parser.add_argument(
        "--test",
        action="store_true")
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test: --stop-reward must "
             "be achieved within --stop-timesteps AND --stop-iters.")

    return parser
