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
from gymnasium_robotics.envs.multiagent_mujoco import MultiAgentMujocoEnv
import numpy as np
import time

# Gymnasium-Robotics based MAMuJoCo example, you can add / customize your own
# referring to https://robotics.farama.org/envs/MaMuJoCo/

env_args_dict = {
    "2AgentAnt": {"scenario": "Ant",  # "Ant-v4"
                  "agent_conf": "2x4",
                  "agent_obsk": 1,
                  "episode_limit": 1000},
    "4AgentAnt": {"scenario": "Ant",
                  "agent_conf": "4x2",
                  "agent_obsk": 1,
                  "episode_limit": 1000},
    "2AgentHalfCheetah": {"scenario": "HalfCheetah",
                          "agent_conf": "2x3",
                          "agent_obsk": 1,
                          "episode_limit": 1000},
    "6AgentHalfCheetah": {"scenario": "HalfCheetah",
                          "agent_conf": "6x1",
                          "agent_obsk": 1,
                          "episode_limit": 1000},
}

policy_mapping_dict = {
    "all_scenario": {
        "description": "mamujoco all scenarios",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


# maintained version of https://github.com/Farama-Foundation/Gymnasium-Robotics
class RLlibGymnasiumRoboticsMAMujoco(MultiAgentEnv):

    def __init__(self, env_config):
        self.env_config = env_args_dict[env_config["map_name"]]
        map_name = env_config["map_name"]
        scenario_name = env_args_dict[map_name]["scenario"]
        partition = env_args_dict[map_name]["agent_conf"]
        agent_obsk = env_args_dict[map_name]["agent_obsk"]
        self.episode_limit = env_args_dict[map_name]["episode_limit"]

        self.env = MultiAgentMujocoEnv(scenario_name, agent_conf=partition, agent_obsk=agent_obsk)
        self.action_space = Box(-1.0, 1.0, shape=(self.env.action_spaces["agent_0"].shape[0],), dtype=np.float32)
        self.state_space = self.env.single_agent_env.observation_space
        self.state_dim = self.state_space.shape[0]

        self.observation_space = GymDict({
            "obs": Box(-10000, 10000.0, shape=(self.env.observation_spaces["agent_0"].shape[0],), dtype=np.float32),
            "state": Box(-10000.0, 10000.0, shape=(self.state_space.shape[0],), dtype=np.float32),
        })

        if "|" in self.env_config["agent_conf"]:
            self.num_agents = len(self.env_config["agent_conf"].split("|"))
        else:
            self.num_agents = int(self.env_config["agent_conf"].split("x")[0])

        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        o = self.env.reset()
        s = self.env.state()
        obs = {}
        for agent_index, agent_name in enumerate(self.agents):
            obs[agent_name] = {
                "obs": np.float32(o[0][agent_name]),
                "state": np.float32(s),
            }
        return obs

    def step(self, action_dict):
        self.step_count += 1
        o, r, d, _, info = self.env.step(action_dict)
        s = self.env.state()
        rewards = {}
        obs = {}
        for agent_index, agent_name in enumerate(self.agents):
            rewards[agent_name] = r[agent_name]
            obs[agent_name] = {
                "obs": np.float32(o[agent_name]),
                "state": np.float32(s),
            }
        dones = {"__all__": False if sum(d.values()) == 0 else True}
        if self.step_count == self.episode_limit:  # terminate:
            dones = {"__all__": True}
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
            "episode_limit": self.episode_limit,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
