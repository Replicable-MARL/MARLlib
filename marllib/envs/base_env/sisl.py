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

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Discrete, Box
import supersuit as ss
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from pettingzoo.sisl import multiwalker_v7, pursuit_v3, waterworld_v3
import time

REGISTRY = {}
REGISTRY["multiwalker"] = multiwalker_v7.parallel_env
REGISTRY["pursuit"] = pursuit_v3.parallel_env
REGISTRY["waterworld"] = waterworld_v3.parallel_env

policy_mapping_dict = {
    "multiwalker": {
        "description": "one team cooperate",
        "team_prefix": ("walker_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    "pursuit": {
        "description": "one team cooperate",
        "team_prefix": ("pursuer_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    "waterworld": {
        "description": "one team cooperate",
        "team_prefix": ("pursuer_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    }
}


class RLlibSISL(MultiAgentEnv):

    def __init__(self, env_config):
        map = env_config["map_name"]
        env_config.pop("map_name", None)
        env = REGISTRY[map](**env_config)

        # keep obs and action dim same across agents
        # pad_action_space_v0 will auto mask the padding actions
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)

        self.env = ParallelPettingZooEnv(env)
        self.action_space = self.env.action_space
        one_agent_name = policy_mapping_dict[map]["team_prefix"][0] + "0"
        self.observation_space = GymDict({"obs": self.env.observation_spaces[one_agent_name]})
        self.agents = self.env.agents
        self.num_agents = len(self.agents)
        env_config["map_name"] = map
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for i in self.agents:
            obs[i] = {"obs": original_obs[i]}
        return obs

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        rewards = {}
        obs = {}
        for key in action_dict.keys():
            rewards[key] = r[key]
            obs[key] = {
                "obs": o[key]
            }
        dones = {"__all__": d["__all__"]}
        return obs, rewards, dones, info

    def close(self):
        self.env.close()

    def render(self, mode=None):
        self.env.render()
        time.sleep(0.05)
        return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env_config["max_cycles"],
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
