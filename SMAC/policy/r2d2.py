from ray import tune
from ray.rllib.agents.dqn.r2d2 import DEFAULT_CONFIG
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.dqn.r2d2_tf_policy import R2D2TFPolicy
from SMAC.metric.smac_callback import *
from SMAC.util.r2d2_tools import *
from SMAC.model.torch_qmix_mask_gru_updet import *


def run_r2d2(args, common_config, env_config, stop):
    # ray built-in Q series algo is not very flexible
    if args.neural_arch not in ["GRU", "LSTM"]:
        assert NotImplementedError

    obs_shape = env_config["obs_shape"]
    n_ally = env_config["n_ally"]
    n_enemy = env_config["n_enemy"]
    state_shape = env_config["state_shape"]
    n_actions = env_config["n_actions"]
    rollout_fragment_length = env_config["rollout_fragment_length"]

    config = {
        "env": "smac",
        "model": {
            "custom_model": "{}_IndependentCritic".format(args.neural_arch),
            "max_seq_len": rollout_fragment_length,
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

    DEFAULT_CONFIG['dueling'] = False
    R2D2WithMaskTrainer = build_trainer(
        name="R2D2_Trainer",
        default_config=DEFAULT_CONFIG,
        default_policy=R2D2TFPolicy,
        get_policy_class=get_policy_class,
    )

    results = tune.run(R2D2WithMaskTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                       config=config,
                       verbose=1)

    return results
