from ray import tune
from ray.tune.utils import merge_dicts
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG
from SMAC.util.coma_tools import *


def run_coma(args, common_config, env_config, stop, reporter):
    obs_shape = env_config["obs_shape"]
    n_ally = env_config["n_ally"]
    n_enemy = env_config["n_enemy"]
    state_shape = env_config["state_shape"]
    n_actions = env_config["n_actions"]
    episode_limit = env_config["episode_limit"]
    episode_num = 10
    train_batch_size = episode_num * episode_limit

    config = {
        "env": "smac",
        "batch_mode": "complete_episodes",
        "train_batch_size": train_batch_size,
        "lr": 0.0005,
        "entropy_coeff": 0.01,
        "model": {
            "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
            "max_seq_len": episode_limit,
            "custom_model_config": {
                "token_dim": args.token_dim,
                "ally_num": n_ally,
                "enemy_num": n_enemy,
                "self_obs_dim": obs_shape,
                "state_dim": state_shape,
                "coma": True
            },
        },
    }

    config.update(common_config)

    COMA_CONFIG = merge_dicts(
        A2C_CONFIG,
        {
            "agent_num": n_ally,
            "state_dim": state_shape,
            "self_obs_dim": obs_shape,
            "centralized_critic_obs_dim": -1,
        }
    )

    # not used
    COMATFPolicy = A3CTFPolicy.with_updates(
        name="COMATFPolicy",
        postprocess_fn=centralized_critic_postprocessing,
        loss_fn=loss_with_central_critic_coma,
        grad_stats_fn=central_vf_stats_coma,
        mixins=[
            CentralizedValueMixin
        ])

    # based on a3c torch policy
    COMATorchPolicy = A3CTorchPolicy.with_updates(
        name="COMATorchPolicy",
        get_default_config=lambda: COMA_CONFIG,
        loss_fn=coma_loss,
        postprocess_fn=centralized_critic_postprocessing_coma,
        extra_action_out_fn=coma_model_value_predictions,
    )

    def get_policy_class(config_):
        if config_["framework"] == "torch":
            return COMATorchPolicy

    COMATrainer = A2CTrainer.with_updates(
        name="COMATrainer",
        default_policy=COMATFPolicy,
        get_policy_class=get_policy_class,
    )

    results = tune.run(COMATrainer,
                       name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop, config=config,
                       verbose=1, progress_reporter=reporter)

    return results
