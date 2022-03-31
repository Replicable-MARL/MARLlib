import numpy as np
import ray
from ray import tune
from config_nmmo import *
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved

from NeuralMMO.env.neural_mmo_rllib import NeuralMMO_RLlib
from NeuralMMO.model.torch_lstm_baseline import NMMO_Baseline_LSTM
from NeuralMMO.model.torch_gru_baseline import NMMO_Baseline_GRU

from NeuralMMO.model.utils.spaces import observationSpace, actionSpace
from NeuralMMO.metric.callback import NMMO_Callbacks

from NeuralMMO.policy.pg_a2c_a3c import run_pg_a2c_a3c
from NeuralMMO.policy.ppo import run_ppo

if __name__ == '__main__':

    args = get_train_parser().parse_args()
    ray.init(local_mode=args.local_mode)

    ###################
    ### environment ###
    ###################

    if args.local_mode:
        customized_game_config = Debug()  # local testing without GPU
    else:
        customized_game_config = CompetitionRound1()

    # Obs and actions
    obs = observationSpace(customized_game_config)
    atns = actionSpace(customized_game_config)

    # Register custom env and policies
    ray.tune.registry.register_env("Neural_MMO",
                                   lambda customized_game_config: NeuralMMO_RLlib(customized_game_config))

    ##############
    ### policy ###
    ##############

    mapPolicy = lambda agentID: 'policy_{}'.format(
        agentID % customized_game_config.NPOLICIES)

    policies = {}

    for i in range(customized_game_config.NPOLICIES):
        params = {
            "agent_id": i,
            "obs_space_dict": obs,
            "act_space_dict": atns}
        key = mapPolicy(i)
        policies[key] = (None, obs, atns, params)

    policy_function_dict = {
        "PG": run_pg_a2c_a3c,
        "A2C": run_pg_a2c_a3c,
        "A3C": run_pg_a2c_a3c,
        "PPO": run_ppo,
    }

    #############
    ### model ###
    #############

    ModelCatalog.register_custom_model('Torch_LSTM_Baseline', NMMO_Baseline_LSTM)
    ModelCatalog.register_custom_model('Torch_GRU_Baseline', NMMO_Baseline_GRU)

    #####################
    ### common config ###
    #####################

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

    stop = {'training_iteration': customized_game_config.TRAINING_ITERATIONS}

    ##################
    ### run script ###
    ###################

    results = policy_function_dict[args.run](args, common_config, customized_game_config, stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
