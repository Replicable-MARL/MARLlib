''' copied and modified from Neural-MMO RLlib_Wrapper '''

from neural_mmo.forge.blade.core import config
from neural_mmo.forge.trinity.scripted import baselines
from neural_mmo.forge.trinity.agent import Agent
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
        choices=["QMIX", "VDN", "R2D2", "PG", "A2C", "A3C", "PPO"],  # "APPO" "IMPALA"
        default="PPO",
        help="The RLlib-registered algorithm to use.")
    parser.add_argument(
        "--neural-arch",
        choices=["LSTM", "GRU"],
        type=str,
        default="GRU",
        help="Agent Neural Architecture")

    return parser


class RLlibConfig:
    '''Base config for RLlib Models
    Extends core Config, which contains environment, evaluation,
    and non-RLlib-specific learning parameters'''

    @property
    def MODEL(self):
        return self.__class__.__name__

    # Checkpointing. Resume will load the latest trial, e.g. to continue training
    # Restore (overrides resume) will force load a specific checkpoint (e.g. for rendering)
    RESUME = False
    RESTORE = 'your_dir_path/checkpoint_001000/checkpoint-1000'

    # Policy specification
    AGENTS = [Agent]
    EVAL_AGENTS = [baselines.Meander, baselines.Forage, baselines.Combat, Agent]
    EVALUATE = False  # Reserved param

    # Hardware and debug
    NUM_WORKERS = 1
    NUM_GPUS_PER_WORKER = 0
    NUM_GPUS = 1
    EVALUATION_NUM_WORKERS = 1
    LOCAL_MODE = False
    LOG_LEVEL = 1

    # Training and evaluation settings
    EVALUATION_INTERVAL = 1
    EVALUATION_NUM_EPISODES = 3
    EVALUATION_PARALLEL = True
    TRAINING_ITERATIONS = 1000
    KEEP_CHECKPOINTS_NUM = 5
    CHECKPOINT_FREQ = 1
    LSTM_BPTT_HORIZON = 16
    NUM_SGD_ITER = 1

    # Model
    SCRIPTED = None
    N_AGENT_OBS = 100
    NPOLICIES = 1
    HIDDEN = 64
    EMBED = 64

    # Reward
    TEAM_SPIRIT = 0.0
    ACHIEVEMENT_SCALE = 1.0 / 15.0


class SmallMaps(RLlibConfig, config.AllGameSystems, config.SmallMaps):
    '''Small scale Neural MMO training setting
    Features up to 128 concurrent agents and 32 concurrent NPCs,
    60x60 maps (excluding the border), and 1000 timestep train/eval horizons.

    This setting is modeled off of v1.1-v1.4 It is appropriate as a quick train
    task for new ideas, a transfer target for agents trained on large maps,
    or as a primary research target for PCG methods.'''

    # Memory/Batch Scale
    NUM_WORKERS = 1
    TRAIN_BATCH_SIZE = 64 * 32 * NUM_WORKERS
    ROLLOUT_FRAGMENT_LENGTH = 256
    SGD_MINIBATCH_SIZE = 128

    # Horizon
    TRAIN_HORIZON = 1024
    EVALUATION_HORIZON = 1024


### Cheapest environment in AICrowd competition.
class CompetitionRound1(config.Achievement, SmallMaps):
    @property
    def SPAWN(self):
        return self.SPAWN_CONCURRENT

    NENT = 128
    NPOP = 1


### Greatly reduced parameters for debugging (used in local mode).
class Debug(SmallMaps, config.AllGameSystems):
    LOAD = False
    LOCAL_MODE = True
    NUM_WORKERS = 1

    SGD_MINIBATCH_SIZE = 100
    TRAIN_BATCH_SIZE = 400
    TRAIN_HORIZON = 200
    EVALUATION_HORIZON = 50

    HIDDEN = 2
    EMBED = 2
