# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import gfootball.env as football_env
import gym
from gym.spaces import Dict as GymDict, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.utils import merge_dicts

SMM_WIDTH = 32
SMM_HEIGHT = 32

# only cooperative scenario
ally_num_dict = {
    "academy_pass_and_shoot_with_keeper": 2,
    "academy_run_pass_and_shoot_with_keeper": 2,
    "academy_3_vs_1_with_keeper": 3,
    "academy_counterattack_easy": 4,
    "academy_counterattack_hard": 4,
    "academy_single_goal_versus_lazy": 11,
}

policy_mapping_dict = {
    "all_scenario": {
        "description": "football all scenarios",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}

class RLlibGFootball(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def __init__(self, env_config):
        env_config["env_name"] = env_config.pop("map_name")
        self.env_config = env_config
        self.num_agents = ally_num_dict[self.env_config["env_name"]]

        extra_setting = {
            "number_of_left_players_agent_controls": self.num_agents,
            "channel_dimensions": (SMM_WIDTH, SMM_HEIGHT),
        }

        self.env = football_env.create_environment(**merge_dicts(self.env_config, extra_setting))
        self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
        self.observation_space = GymDict({"obs": Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            dtype=self.env.observation_space.dtype)})
        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]

        # back to be compatible in run script
        env_config["map_name"] = env_config.pop("env_name")

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for x in range(self.num_agents):
            obs["agent_%d" % x] = {
                "obs": original_obs[x]
            }
        return obs

    def step(self, action_dict):
        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)
        o, r, d, i = self.env.step(actions)
        rewards = {}
        obs = {}
        infos = {}
        for pos, key in enumerate(sorted(action_dict.keys())):
            infos[key] = i
            rewards[key] = r[pos]
            obs[key] = {
                "obs": o[pos]
            }
        dones = {"__all__": d}
        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 400,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
