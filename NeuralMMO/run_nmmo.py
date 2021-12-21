import numpy as np
import ray
from ray import tune

from config_nmmo import *
from ray.rllib.models import ModelCatalog
from NeuralMMO.env.neural_mmo_rllib import NeuralMMO_RLlib
from NeuralMMO.model.torch_lstm_baseline import NMMO_Baseline_LSTM
from NeuralMMO.model.torch_gru_baseline import NMMO_Baseline_GRU

from NeuralMMO.model.utils.spaces import observationSpace, actionSpace
from NeuralMMO.metric.callback import NMMO_Callbacks
from NeuralMMO.trainer.ppo_trainer import Customized_PPOTrainer as PPOTrainer
from NeuralMMO.trainer.a2c_trainer import Customized_A2CTrainer as A2CTrainer
from NeuralMMO.trainer.a3c_trainer import Customized_A3CTrainer as A3CTrainer
from NeuralMMO.trainer.pg_trainer import Customized_PGTrainer as PGTrainer

import sys
# from ray.rllib.agents.ppo.ppo import PPOTrainer

if __name__ == '__main__':

    args = get_train_parser().parse_args()

    ray.init(local_mode=args.local_mode)

    if args.local_mode:
        customized_game_config = Debug()  # local testing without GPU
    else:
        customized_game_config = CompetitionRound1()

    # Register custom env and policies
    ray.tune.registry.register_env("Neural_MMO",
                                   lambda customized_game_config: NeuralMMO_RLlib(customized_game_config))
    ModelCatalog.register_custom_model('Torch_LSTM_Baseline', NMMO_Baseline_LSTM)
    ModelCatalog.register_custom_model('Torch_GRU_Baseline', NMMO_Baseline_GRU)

    mapPolicy = lambda agentID: 'policy_{}'.format(
        agentID % customized_game_config.NPOLICIES)

    policies = {}
    # Obs and actions
    obs = observationSpace(customized_game_config)
    atns = actionSpace(customized_game_config)
    for i in range(customized_game_config.NPOLICIES):
        params = {
            "agent_id": i,
            "obs_space_dict": obs,
            "act_space_dict": atns}
        key = mapPolicy(i)
        policies[key] = (None, obs, atns, params)

    # Create rllib config
    common_config = {
        'num_workers': customized_game_config.NUM_WORKERS,
        'num_gpus_per_worker': customized_game_config.NUM_GPUS_PER_WORKER,
        'num_gpus': customized_game_config.NUM_GPUS,
        'num_envs_per_worker': 1,
        'train_batch_size': customized_game_config.TRAIN_BATCH_SIZE,
        'rollout_fragment_length': customized_game_config.ROLLOUT_FRAGMENT_LENGTH,
        'framework': 'torch',
        'horizon': np.inf,
        'soft_horizon': False,
        'no_done_at_end': False,
        'env': 'Neural_MMO',
        'env_config': {
            'config': customized_game_config
        },
        'multiagent': {
            'policies': policies,
            'policy_mapping_fn': mapPolicy,
            'count_steps_by': 'agent_steps'
        },
        'callbacks': NMMO_Callbacks,
    }

    name = customized_game_config.__class__.__name__ + "_" + args.run + "_" + args.neural_arch

    if args.run in ["QMIX", "VDN"]:
        assert NotImplementedError
        print("Neural-MMO is competitive/cooperative mixing setting environment"
              "\n Joint Q learning algos like not QMIX and VDN are not suitable under this setting")
        sys.exit()

    elif args.run in ["R2D2"]:
        assert NotImplementedError
        print("Action space (MultiAction) of Neural-MMO is not supported for Q function based algo like DQN and R2D2 in Ray/RLlib")
        sys.exit()

    elif args.run in ["PG", "A2C", "A3C"]:
        config = {
            'model': {
                'custom_model': 'Torch_{}_Baseline'.format(args.neural_arch),
                'custom_model_config': {'config': customized_game_config},
                'max_seq_len': customized_game_config.LSTM_BPTT_HORIZON
            },
        }

        config.update(common_config)

        if args.run == "PG":
            trainer = PGTrainer
        elif args.run == "A2C":
            trainer = A2CTrainer
        else:  # A3C
            trainer = A3CTrainer

        tune.run(trainer,  # trainer provided by Neural-MMO repo
                 config=config,
                 name=name,
                 stop={'training_iteration': customized_game_config.TRAINING_ITERATIONS},
                 verbose=customized_game_config.LOG_LEVEL)


    elif args.run in ["PPO"]:
        config = {
            'model': {
                'custom_model': 'Torch_{}_Baseline'.format(args.neural_arch),
                'custom_model_config': {'config': customized_game_config},
                'max_seq_len': customized_game_config.LSTM_BPTT_HORIZON
            },
            'num_sgd_iter': customized_game_config.NUM_SGD_ITER,
            'sgd_minibatch_size': customized_game_config.SGD_MINIBATCH_SIZE,
        }

        config.update(common_config)

        tune.run(PPOTrainer,  # trainer provided by Neural-MMO repo
                 config=config,
                 name=name,
                 stop={'training_iteration': customized_game_config.TRAINING_ITERATIONS},
                 verbose=customized_game_config.LOG_LEVEL)
