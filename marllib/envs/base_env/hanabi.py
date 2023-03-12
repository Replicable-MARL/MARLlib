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
from marllib.patch.hanabi.Hanabi_Env import HanabiEnv
import numpy as np
from gym.spaces import Dict as GymDict, Discrete, Box

policy_mapping_dict = {
    "all_scenario": {
        "description": "hanabi all scenarios",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


class RLlibHanabi(MultiAgentEnv):

    def __init__(self, env_config):
        self.env = HanabiEnv(env_config, env_config["seed"])

        self.action_space = self.env.action_space[0]
        self.observation_space = GymDict({
            "obs": Box(-100.0, 100.0, shape=(self.env.observation_space[0][0],), ),
            "state": Box(-100.0, 100.0, shape=(self.env.share_observation_space[0][0],), ),
            "action_mask": Box(-100.0, 100.0, shape=(self.env.action_space[0].n,)),
        })
        self.num_agents = env_config["num_agents"]
        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        self.env_config = env_config

    def reset(self):
        o, s, action_mask = self.env.reset()
        agent_flag = list(s[-self.num_agents:]).index(1)
        obs = {}
        obs["agent_{}".format(agent_flag)] = {
            "obs": np.float32(np.array(o)),
            "state": np.float32(s),
            "action_mask": np.float32(action_mask)
        }
        return obs

    def step(self, action_dict):
        action = action_dict[next(iter(action_dict))]
        o, s, r, d, info, action_mask = self.env.step([action])
        agent_flag = list(s[-self.num_agents:]).index(1)
        rewards = {}
        obs = {}
        rewards["agent_{}".format(agent_flag)] = r[agent_flag][0]
        obs["agent_{}".format(agent_flag)] = {
            "obs": np.float32(np.array(o)),
            "state": np.float32(s),
            "action_mask": np.float32(action_mask)
        }
        dones = {"__all__": d}
        return obs, rewards, dones, {}

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
