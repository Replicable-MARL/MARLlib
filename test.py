# See also: centralized_critic.py for centralized critic PPO on twp step game.

import os
import ray

from gym.spaces import Dict as GymDict

from ray import tune
from ray.tune import register_env
from ray.tune.utils import merge_dicts

from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models import ModelCatalog
from ray.rllib.env.multi_agent_env import ENV_STATE

from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.a3c.a3c import DEFAULT_CONFIG as A3C_CONFIG

from model.torch_mask_lstm import *
from model.torch_mask_lstm_cc import *
from model.torch_mask_updet import *
from model.torch_mask_updet_cc import *
from marl_util.mappo_tools import *
from marl_util.maa2c_tools import *
from metric.smac_callback import *

# from smac.env.starcraft2.starcraft2 import StarCraft2Env as SMAC
from env.starcraft2_rllib import StarCraft2Env_Rllib as SMAC
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole

if __name__ == '__main__':
    ray.init(local_mode=True)

    stop = {
        "episode_reward_mean": 150,
        "timesteps_total": 1000000,
    }
    config = {
        "env": StatelessCartPole,
        "framework": tf,
        "num_workers": 0,
        # R2D2 settings.
        "burn_in": 20,
        "zero_init_states": True,
        # dueling: false
        "lr": 0.0005,
        # Give some more time to explore.
        "exploration_config": {
            "epsilon_timesteps": 50000,
        },
        # Wrap with an LSTM and use a very simple base-model.
        "model": {
            "fcnet_hiddens": [64],
            "fcnet_activation": "linear",
            "use_lstm": True,
            "lstm_cell_size": 64,
            "max_seq_len": 20,
        }
    }
    tune.run("R2D2", stop=stop, config=config)

    ray.shutdown()
