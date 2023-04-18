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


policy_mapping_dict = {
    "all_scenario": {
        "description": "mate mixed scenarios",
        "team_prefix": ("camera_", "target_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    },
}

class RLlibMATE(MultiAgentEnv):

    def __init__(self, env_config):
        map = env_config["map_name"]
        env_config.pop("map_name", None)
        self.env = mate.make(map)
        if not env_config["continuous_actions_camera"]:
            self.env = mate.DiscreteCamera(self.env, levels=env_config["discrete_levels"])
        if not env_config["continuous_actions_target"]:
            self.env = mate.DiscreteTarget(self.env, levels=env_config["discrete_levels"])
        self.action_space_camera = self.env.action_space.spaces[0].spaces[0]
        self.action_space_target = self.env.action_space.spaces[1].spaces[0]

        self.observation_space_camera = GymDict({"obs": self.env.observation_space.spaces[0].spaces[0]})
        self.observation_space_target = GymDict({"obs": self.env.observation_space.spaces[1].spaces[0]})

        # for gym/rllib usage compatible purpose; not functioning in real use
        self.observation_space = self.observation_space_camera
        self.action_space = self.action_space_camera

        self.agents_camera = ["camera_{}".format(i) for i in range(len(self.env.cameras))]
        self.agents_target = ["target_{}".format(i) for i in range(len(self.env.targets))]

        self.agents = self.agents_camera + self.agents_target
        self.num_agents = len(self.agents)
        env_config["map_name"] = map
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for i, name in enumerate(self.agents_camera):
            obs[name] = {"obs": original_obs[0][i]}
        for i, name in enumerate(self.agents_target):
            obs[name] = {"obs": original_obs[1][i]}
        return obs

    def step(self, action_dict):
        action_camera = []
        for camera_name in self.agents_camera:
            action_camera.append(action_dict[camera_name])
        action_camera = tuple(action_camera)
        action_target = []
        for target_name in self.agents_target:
            action_target.append(action_dict[target_name])
        action_target = tuple(action_target)

        (
            (camera_joint_observation, target_joint_observation),
            (camera_team_reward, target_team_reward),
            done,
            (camera_infos, target_infos)
        ) = self.env.step((action_camera, action_target))

        rewards = {}
        obs = {}

        for i, name in enumerate(self.agents_camera):
            rewards[name] = camera_team_reward
            obs[name] = {
                "obs": camera_joint_observation[i]
            }

        for i, name in enumerate(self.agents_target):
            rewards[name] = target_team_reward
            obs[name] = {
                "obs": target_joint_observation[i]
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
            "space_obs_camera": self.observation_space_camera,
            "space_act_camera": self.action_space_camera,
            "space_obs_target": self.observation_space_target,
            "space_act_target": self.action_space_target,
            "num_agents": self.num_agents,
            "agents": self.agents,
            "episode_limit": 2000,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
