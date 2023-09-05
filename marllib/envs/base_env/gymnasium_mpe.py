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
import time

# pettingzoo 1.23.0
from pettingzoo.mpe import simple_adversary_v3, simple_crypto_v3, simple_push_v3, simple_spread_v3, simple_tag_v3, \
    simple_reference_v3, simple_world_comm_v3, simple_speaker_listener_v4

REGISTRY = {}
REGISTRY["simple_adversary"] = simple_adversary_v3.parallel_env
REGISTRY["simple_crypto"] = simple_crypto_v3.parallel_env
REGISTRY["simple_push"] = simple_push_v3.parallel_env
REGISTRY["simple_tag"] = simple_tag_v3.parallel_env
REGISTRY["simple_spread"] = simple_spread_v3.parallel_env
REGISTRY["simple_reference"] = simple_reference_v3.parallel_env
REGISTRY["simple_world_comm"] = simple_world_comm_v3.parallel_env
REGISTRY["simple_speaker_listener"] = simple_speaker_listener_v4.parallel_env


policy_mapping_dict = {
    "simple_adversary": {
        "description": "one team attack, one team survive",
        "team_prefix": ("adversary_", "agent_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    },
    "simple_crypto": {
        "description": "two team cooperate, one team attack",
        "team_prefix": ("eve_", "bob_", "alice_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    },
    "simple_push": {
        "description": "one team target on landmark, one team attack",
        "team_prefix": ("adversary_", "agent_",),
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    },
    "simple_tag": {
        "description": "one team attack, one team survive",
        "team_prefix": ("adversary_", "agent_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    },
    "simple_spread": {
        "description": "one team cooperate",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    "simple_reference": {
        "description": "one team cooperate",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    "simple_world_comm": {
        "description": "two team cooperate and attack, one team survive",
        "team_prefix": ("adversary_", "leadadversary_", "agent_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    },
    "simple_speaker_listener": {
        "description": "two team cooperate",
        "team_prefix": ("speaker_", "listener_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


class RLlibMPE_Gymnasium(MultiAgentEnv):

    def __init__(self, env_config):
        map = env_config["map_name"]
        env_config.pop("map_name", None)
        env = REGISTRY[map](**env_config)

        # keep obs and action dim same across agents
        # pad_action_space_v0 will auto mask the padding actions
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)

        self.env = env
        self.action_space = Discrete(self.env.action_spaces.popitem()[1].n)
        self.observation_space = GymDict({"obs": Box(
            low=-100.0,
            high=100.0,
            shape=(self.env.observation_spaces.popitem()[1].shape[0],),
            dtype=self.env.observation_spaces.popitem()[1].dtype)})
        self.agents = self.env.possible_agents
        self.num_agents = len(self.agents)
        env_config["map_name"] = map
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for i in self.agents:
            obs[i] = {"obs": original_obs[0][i]}
        return obs

    def step(self, action_dict):
        o, r, d, t, info = self.env.step(action_dict)
        rewards = {}
        obs = {}
        for agent in self.agents:
            rewards[agent] = r[agent]
            obs[agent] = {
                "obs": o[agent]
            }
        dones = {"__all__": d.popitem()[1] or t.popitem()[1]}
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
            "episode_limit": 25,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
