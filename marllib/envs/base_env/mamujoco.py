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
from multiagent_mujoco.mujoco_multi import MujocoMulti
from gym.spaces import Dict as GymDict, Discrete, Box
import numpy as np
import time
env_args_dict = {
    "2AgentAnt": {"scenario": "Ant-v2",
                  "agent_conf": "2x4",
                  "agent_obsk": 1,
                  "episode_limit": 1000},
    "2AgentAntDiag": {"scenario": "Ant-v2",
                      "agent_conf": "2x4d",
                      "agent_obsk": 1,
                      "episode_limit": 1000},
    "4AgentAnt": {"scenario": "Ant-v2",
                  "agent_conf": "4x2",
                  "agent_obsk": 1,
                  "episode_limit": 1000},
    "2AgentHalfCheetah": {"scenario": "HalfCheetah-v2",
                          "agent_conf": "2x3",
                          "agent_obsk": 1,
                          "episode_limit": 1000},
    "6AgentHalfCheetah": {"scenario": "HalfCheetah-v2",
                          "agent_conf": "6x1",
                          "agent_obsk": 1,
                          "episode_limit": 1000},
    "3AgentHopper": {"scenario": "Hopper-v2",
                     "agent_conf": "3x1",
                     "agent_obsk": 0,
                     "episode_limit": 1000},
    "2AgentHumanoid": {"scenario": "Humanoid-v2",
                       "agent_conf": "9|8",
                       "agent_obsk": 1,
                       "episode_limit": 1000},
    "2AgentHumanoidStandup": {"scenario": "HumanoidStandup-v2",
                              "agent_conf": "9|8",
                              "agent_obsk": 1,
                              "episode_limit": 1000},
    "2AgentReacher": {"scenario": "Reacher-v2",
                      "agent_conf": "2x1",
                      "agent_obsk": 1,
                      "episode_limit": 1000},
    "2AgentSwimmer": {"scenario": "Swimmer-v2",
                      "agent_conf": "2x1",
                      "agent_obsk": 1,
                      "episode_limit": 1000},
    "2AgentWalker": {"scenario": "Walker2d-v2",
                     "agent_conf": "2x3",
                     "agent_obsk": 1,
                     "episode_limit": 1000},
    "ManyagentSwimmer": {"scenario": "manyagent_swimmer",
                         "agent_conf": "10x2",
                         "agent_obsk": 1,
                         "episode_limit": 1000},
    "ManyagentAnt": {"scenario": "manyagent_ant",
                     "agent_conf": "2x3",
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


class RLlibMAMujoco(MultiAgentEnv):

    def __init__(self, env_config):
        self.env_config = env_args_dict[env_config["map_name"]]
        self.env = MujocoMulti(env_args=self.env_config)
        self.action_space = self.env.action_space[0]
        self.state_dim = self.env.wrapped_env.observation_space.shape[0]
        self.observation_space = GymDict({
            "obs": Box(-10000.0, 10000.0, shape=(self.env.obs_size,), dtype=self.env.observation_space[0].dtype),
            "state": Box(-100.0, 100.0, shape=(self.state_dim,), dtype=self.env.observation_space[0].dtype),
        })

        if "|" in self.env_config["agent_conf"]:
            self.num_agents = len(self.env_config["agent_conf"].split("|"))
        else:
            self.num_agents = int(self.env_config["agent_conf"].split("x")[0])

        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]


    def reset(self):
        self.env.reset()
        o = self.env.get_obs()  # obs
        s = self.env.get_state()  # g state
        # to float32 for RLLIB check
        obs = {}
        for agent_index, agent_name in enumerate(self.agents):
            obs[agent_name] = {
                "obs": np.float32(o[agent_index]),
                "state": np.float32(s),
            }
        return obs

    def step(self, action_dict):
        # print(f"Running Env ID: {id(self)}")
        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)

        actions = normalize_action(np.array(actions), self.action_space)

        r, d, _ = self.env.step(actions)

        o = self.env.get_obs()  # obs
        s = self.env.get_state()  # g state

        o = normalize_obs(o)

        rewards = {}
        obs = {}
        infos = {}
        # to float32 for RLLIB check
        for pos, key in enumerate(sorted(action_dict.keys())):
            rewards[key] = r
            obs[key] = o[pos]
            obs[key] = {
                "obs": np.float32(o[pos]),
                "state": np.float32(s),
            }
        dones = {"__all__": d}
        return obs, rewards, dones, infos

    def close(self):
        pass

    def render(self, mode='human'):
        self.env.render()
        time.sleep(0.05)
        return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 1000,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info


def normalize_obs(obs):
    obs = (obs - np.mean(obs)) / np.std(obs)
    return obs


def normalize_action(action, action_space):
    action = (action + 1) / 2
    action *= (action_space.high - action_space.low)
    action += action_space.low
    return action
