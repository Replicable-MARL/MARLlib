from ray import tune
from NeuralMMO.trainer.ppo_trainer import Customized_PPOTrainer as PPOTrainer

def run_ppo(args, common_config, customized_game_config, stop):
    """
            for bug mentioned https://github.com/ray-project/ray/pull/20743
            make sure sgd_minibatch_size > max_seq_len
            """
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

    name = customized_game_config.__class__.__name__ + "_" + args.run + "_" + args.neural_arch

    results = tune.run(PPOTrainer,  # trainer provided by Neural-MMO repo
                       config=config,
                       name=name,
                       stop={'training_iteration': customized_game_config.TRAINING_ITERATIONS},
                       verbose=customized_game_config.LOG_LEVEL)

    return results
