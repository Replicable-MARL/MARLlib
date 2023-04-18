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
from gym.spaces import Dict as GymDict, Box
import supersuit as ss
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from pettingzoo.magent import adversarial_pursuit_v3, battle_v3, battlefield_v3, combined_arms_v5, gather_v3, \
    tiger_deer_v3

REGISTRY = {}
REGISTRY["adversarial_pursuit"] = adversarial_pursuit_v3.env
REGISTRY["battle"] = battle_v3.env
REGISTRY["battlefield"] = battlefield_v3.env
REGISTRY["combined_arms"] = combined_arms_v5.env
REGISTRY["gather"] = gather_v3.env
REGISTRY["tiger_deer"] = tiger_deer_v3.env

mini_channel_dim_dict = {
    "adversarial_pursuit": 4,
    "battle": 4,
    "battlefield": 4,
    "combined_arms": 6,
    "gather": 4,
    "tiger_deer": 6,
}

# magent agent number is large, one_agent_one_policy is set to be False on all scenarios
policy_mapping_dict = {
    "adversarial_pursuit": {
        "description": "one team attack, one team survive",
        "team_prefix": ("predator_", "prey_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": False,
    },
    "battle": {
        "description": "two team battle",
        "team_prefix": ("red_", "blue_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": False,
    },
    "battlefield": {
        "description": "two team battle",
        "team_prefix": ("red_", "blue_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": False,
    },
    "combined_arms": {
        "description": "two team battle with mixed type of units",
        "team_prefix": ("redranged_", "redmelee_", "bluemele_", "blueranged_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": False,
    },
    "gather": {
        "description": "survive",
        "team_prefix": ("omnivore_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": False,
    },
    "tiger_deer": {
        "description": "one team attack, one team survive",
        "team_prefix": ("tiger_", "deer_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": False,
    },
}


class RLlibMAgent(MultiAgentEnv):

    def __init__(self, env_config):
        map = env_config.pop("map_name", None)

        env = REGISTRY[map](**env_config)

        # keep obs and action dim same across agents
        # pad_action_space_v0 will auto mask the padding actions
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)

        self.env = PettingZooEnv(env)
        self.mini_channel_dim = mini_channel_dim_dict[map]
        self.action_space = self.env.action_space
        self.observation_space = GymDict({
            "obs": Box(
                low=self.env.observation_space.low[:, :, :-self.mini_channel_dim],
                high=self.env.observation_space.high[:, :, :-self.mini_channel_dim],
                dtype=self.env.observation_space.dtype),
            "state": Box(
                low=self.env.observation_space.low[:, :, -self.mini_channel_dim:],
                high=self.env.observation_space.high[:, :, -self.mini_channel_dim:],
                dtype=self.env.observation_space.dtype),
        })
        self.agents = self.env.agents
        self.num_agents = len(self.agents)
        env_config["map_name"] = map
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for key in original_obs.keys():
            obs[key] = {
                "obs": original_obs[key][:, :, :-self.mini_channel_dim],
                "state": original_obs[key][:, :, -self.mini_channel_dim:]
            }
        return obs

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        rewards = {}
        obs = {}
        for key in o.keys():
            rewards[key] = r[key]
            obs[key] = {
                "obs": o[key][:, :, :-self.mini_channel_dim],
                "state": o[key][:, :, -self.mini_channel_dim:]
            }
        dones = {"__all__": d["__all__"]}
        return obs, rewards, dones, info

    def close(self):
        self.env.close()

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 200,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
