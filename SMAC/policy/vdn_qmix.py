from gym.spaces import Dict as GymDict
from ray import tune
from ray.tune import register_env
from SMAC.model.torch_mask_updet_cc import *
from SMAC.util.r2d2_tools import *
from SMAC.model.torch_qmix_mask_gru_updet import *

from SMAC.env.starcraft2_rllib import StarCraft2Env_Rllib as SMAC
import os


def run_vdn_qmix(args, common_config, env_config, stop):
    if args.neural_arch not in ["GRU", "UPDeT"]:
        assert NotImplementedError

    obs_shape = env_config["obs_shape"]
    n_ally = env_config["n_ally"]
    n_enemy = env_config["n_enemy"]
    state_shape = env_config["state_shape"]
    n_actions = env_config["n_actions"]
    rollout_fragment_length = env_config["rollout_fragment_length"]

    grouping = {
        "group_1": [i for i in range(n_ally)],
    }
    ## obs state setting here
    obs_space = Tuple([
                          GymDict({
                              "obs": Box(-2.0, 2.0, shape=(obs_shape,)),
                              "state": Box(-2.0, 2.0, shape=(state_shape,)),
                              "action_mask": Box(0.0, 1.0, shape=(n_actions,))
                          })] * n_ally
                      )
    act_space = Tuple([
                          Discrete(n_actions)
                      ] * n_ally)

    # QMIX/VDN need grouping
    register_env(
        "grouped_smac",
        lambda config: SMAC(config).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space))

    config = {
        "env": "grouped_smac",
        "env_config": {
            "map_name": args.map,
        },
        "rollout_fragment_length": rollout_fragment_length,
        "train_batch_size": 400,
        "exploration_config": {
            "epsilon_timesteps": 50000,
            "final_epsilon": 0.05,
        },
        "model": {
            "custom_model_config": {
                "neural_arch": args.neural_arch,
                "token_dim": args.token_dim,
                "ally_num": n_ally,
                "enemy_num": n_enemy,
                "self_obs_dim": obs_shape,
                "state_dim": state_shape
            },
        },
        "mixer": "qmix" if args.run == "QMIX" else None,  # VDN has no mixer network

        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "1")),
        "num_workers": args.num_workers,
    }

    results = tune.run(QMixTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                       config=config, verbose=1)

    return results
