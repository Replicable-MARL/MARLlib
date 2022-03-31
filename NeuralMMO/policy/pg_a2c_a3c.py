from ray import tune
from NeuralMMO.trainer.a2c_trainer import Customized_A2CTrainer as A2CTrainer
from NeuralMMO.trainer.a3c_trainer import Customized_A3CTrainer as A3CTrainer
from NeuralMMO.trainer.pg_trainer import Customized_PGTrainer as PGTrainer


def run_pg_a2c_a3c(args, common_config, customized_game_config, stop):
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

    name = customized_game_config.__class__.__name__ + "_" + args.run + "_" + args.neural_arch

    results = tune.run(trainer,  # trainer provided by Neural-MMO repo
                       config=config,
                       name=name,
                       stop=stop,
                       verbose=customized_game_config.LOG_LEVEL)

    return results
