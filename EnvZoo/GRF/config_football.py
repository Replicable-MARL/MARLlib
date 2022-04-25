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
        "--parallel",
        default=False,
        type=bool,
        help="Whether use tune grid research")
    parser.add_argument(
        "--run",
        choices=["IQL", "QMIX", "VDN", "R2D2", "PG", "A2C", "A3C", "MAA2C", "PPO", "MAPPO", "COMA", "SUM-VDA2C", "MIX-VDA2C", "SUM-VDPPO", "MIX-VDPPO"],  # "APPO" "IMPALA"
        default="IQL",
        help="The RLlib-registered algorithm to use.")
    parser.add_argument(
        "--map",
        choices=[
            "academy_3_vs_1_with_keeper",  # 4 ally left side, 2 opponent right side
            "academy_empty_goal",  # 2 ally left side, 1 opponent right side
            "academy_empty_goal_close",  # 2 ally left side, 1 opponent right side
            "academy_pass_and_shoot_with_keeper",  # 3 ally left side, 2 opponent right side
            "academy_run_pass_and_shoot_with_keeper",  # 3 ally left side, 2 opponent right side
            "academy_run_to_score",  # 2 ally left side, 6 opponent right side
            "academy_run_to_score_with_keeper",  # 2 ally left side, 6 opponent right side
            "academy_single_goal_versus_lazy",  # 11 ally left side, 11 opponent right side
            "academy_corner",  # 11 ally left side, 11 opponent right side
            "academy_counterattack_easy",  # 11 ally left side, 11 opponent right side
            "academy_counterattack_hard",  # 11 ally left side, 11 opponent right side
        ],
        default="academy_3_vs_1_with_keeper",
        help="Envs should be registered")
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=200,
        help="episode-limit for each game")
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
        "--evaluation-interval",
        type=int,
        default=10,
        help="evaluation_interval")
    parser.add_argument(
        "--neural-arch",
        choices=["CNN", "CNN_LSTM", "CNN_GRU", "CNN_UPDeT"],
        type=str,
        default="CNN_GRU",
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
