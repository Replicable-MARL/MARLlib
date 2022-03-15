import argparse


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-mode",
        default=False,
        type=bool,
        help="Init Ray in local mode for easier debugging.")
    parser.add_argument(
        "--parallel",
        default=False,
        type=bool,
        help="Whether use tune grid research")
    parser.add_argument(
        "--run",
        choices=["QMIX", "VDN", "R2D2", "PG", "A2C", "A3C", "MAA2C", "PPO", "MAPPO", "COMA"],  # "APPO" "IMPALA"
        default="COMA",
        help="The RLlib-registered algorithm to use.")
    parser.add_argument(
        "--map",
        choices=[
            "OneVsOne-v0",
            "PommeFFACompetition-v0",
            "PommeTeamCompetition-v0",
            # following scenarios are not tested
            # "PommeFFACompetitionFast-v0",
            # "PommeFFAFast-v0",
            # "PommeFFA-v1",
            # "PommeTeamCompetitionFast-v0",
            # "PommeTeamCompetition-v1",
            # "PommeTeam-v0",
            # "PommeTeamFast-v0",
            # "PommeRadio-v2",
        ],
        default="PommeTeamCompetition-v0",
        help="Envs should be registered")
    parser.add_argument(
        "--agent-position",
        type=str,
        default="0123",
        # choices=["0", "1", "01"] for OneVsOne
        # choices=["0", "1", "2", "3"] random combination for PommeFFACompetition like "023"
        # choices=["01", "23", "0123"] for PommeTeamCompetition
        help="Built-in AI for initialization")
    parser.add_argument(
        "--builtin-ai-type",
        choices=["human_rule", "random_rule"],
        type=str,
        default="human_rule",
        help="Built-in AI for initialization")
    parser.add_argument(
        "--neural-arch",
        choices=["CNN", "CNN_LSTM", "CNN_GRU"],
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
