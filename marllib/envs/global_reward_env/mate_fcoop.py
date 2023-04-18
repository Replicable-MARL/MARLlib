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
import time
import mate
import numpy as np

policy_mapping_dict = {
    "all_scenario": {
        "description": "mate single team multi-agent scenarios",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


class RLlibMATE_FCOOP(MultiAgentEnv):

    def __init__(self, env_config):
        map = env_config["map_name"]
        env_config.pop("map_name", None)
        self.env = mate.make(map)

        if env_config["coop_team"] == "camera":
            if not env_config["continuous_actions_camera"]:
                self.env = mate.DiscreteCamera(self.env, levels=env_config["discrete_levels"])
            self.env = mate.MultiCamera(self.env, target_agent=mate.agents.GreedyTargetAgent(seed=0))
        else:  # target
            if not env_config["continuous_actions_target"]:
                self.env = mate.DiscreteTarget(self.env, levels=env_config["discrete_levels"])
            self.env = mate.MultiTarget(self.env, camera_agent=mate.agents.HeuristicCameraAgent(seed=0))

        self.action_space = self.env.action_space.spaces[0]
        self.observation_space = GymDict({"obs": self.env.observation_space.spaces[0]})
        self.num_agents = self.env.num_teammates
        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        env_config["map_name"] = map
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for i, name in enumerate(self.agents):
            obs[name] = {"obs": original_obs[i]}
        return obs

    def step(self, action_dict):
        action = []
        for name in self.agents:
            action.append(action_dict[name])

        joint_observation, team_reward, done, infos = self.env.step(np.array(action))

        rewards = {}
        obs = {}

        for i, name in enumerate(self.agents):
            rewards[name] = team_reward
            obs[name] = {
                "obs": joint_observation[i]
            }

        dones = {"__all__": done}
        return obs, rewards, dones, {}

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
            "episode_limit": 2000,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
