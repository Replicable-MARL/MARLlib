from ray import tune
from ray.rllib.agents.dqn.r2d2 import DEFAULT_CONFIG as R2D2_CONFIG, R2D2Trainer
from ray.rllib.agents.dqn.r2d2_tf_policy import R2D2TFPolicy
from SMAC.util.r2d2_tools import *
from SMAC.util.qmix_tools import execution_plan_qmix


def run_r2d2(args, common_config, env_config, stop, reporter):
    # ray built-in Q series algo is not very flexible
    if args.neural_arch not in ["GRU", "LSTM"]:
        assert NotImplementedError

    obs_shape = env_config["obs_shape"]
    n_ally = env_config["n_ally"]
    n_enemy = env_config["n_enemy"]
    state_shape = env_config["state_shape"]
    n_actions = env_config["n_actions"]
    episode_limit = env_config["episode_limit"]

    config = {
        "env": "smac",
        "model": {
            "custom_model": "{}_IndependentCritic".format(args.neural_arch),
            "max_seq_len": episode_limit + 1,
            "custom_model_config": {
                "ally_num": n_ally,
                "enemy_num": n_enemy,
                "self_obs_dim": obs_shape,
                "state_dim": state_shape
            },
        },
    }

    config.update(common_config)

    def get_policy_class(config_):
        if config_["framework"] == "torch":
            return R2D2WithMaskPolicy

    learning_starts = episode_limit * 32
    train_batch_size = 32

    R2D2_CONFIG.update(
        {
            "rollout_fragment_length": 1,
            "buffer_size": 5000 * episode_limit // 2,  # in timesteps
            "train_batch_size": train_batch_size,  # in sequence
            "target_network_update_freq": episode_limit * 100,  # in timesteps
            "learning_starts": learning_starts,
            "double_q": True,
            "dueling": False,
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.05,
                "epsilon_timesteps": 50000,  # Timesteps over which to anneal epsilon.
            },
        })

    R2D2WithMaskTrainer = R2D2Trainer.with_updates(
        name="R2D2_Trainer",
        default_config=R2D2_CONFIG,
        default_policy=R2D2TFPolicy,
        get_policy_class=get_policy_class,
        validate_config=r2d2_avoid_bug_validate_config,
        execution_plan=execution_plan_qmix  # use qmix execution plan for fair comparison
    )

    results = tune.run(R2D2WithMaskTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                       config=config,
                       verbose=1,
                       progress_reporter=reporter
                       )

    return results
