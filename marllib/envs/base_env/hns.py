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

from gym.spaces import Dict as GymDict, Box, MultiDiscrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.utils import merge_dicts
from functools import reduce

import numpy as np
from marllib.patch.hns import BlueprintConstructionEnv, BoxLockingEnv, ShelterConstructionEnv, HideAndSeekEnv

policy_mapping_dict = {
    "all_scenario": {
        "description": "hide and seek all scenarios",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    },
}


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class RLlibHideAndSeek(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def __init__(self, env_config):
        map_name = env_config.pop("map_name")
        env_config_obj = Struct(**env_config)
        if map_name == "BlueprintConstruction":
            env = BlueprintConstructionEnv(env_config_obj)
        elif map_name == "BoxLocking":
            env = BoxLockingEnv(env_config_obj)
        elif map_name == "ShelterConstruction":
            env = ShelterConstructionEnv(env_config_obj)
        elif map_name == "hidenseek":
            env = HideAndSeekEnv(env_config_obj)
        else:
            print("Can not support the " + map_name + "environment.")
            raise NotImplementedError
        env.seed(env_config["seed"])

        action_movement = env.action_space.spaces['action_movement'][0].nvec
        action_glueall = env.action_space.spaces['action_glueall'][0].n
        action_vec = np.append(action_movement, action_glueall)
        if 'action_pull' in env.action_space.spaces.keys():
            action_pull = env.action_space.spaces['action_pull'][0].n
            action_vec = np.append(action_vec, action_pull)
        action_space = MultiDiscrete([vec for vec in action_vec])

        # deal with dict obs space
        order_obs = ['agent_qpos_qvel', 'box_obs', 'ramp_obs', 'construction_site_obs', 'observation_self']
        mask_order_obs = [None, None, None, None, None]
        obs_space = []
        obs_dim = 0
        for key in order_obs:
            if key in env.observation_space.spaces.keys():
                space = list(env.observation_space[key].shape)
                if len(space) < 2:
                    space.insert(0, 1)
                obs_space.append(space)
                obs_dim += reduce(lambda x, y: x * y, space)

        self.ori_action_space = env.action_space.spaces
        self.order_obs_key = order_obs
        self.env_config = env_config
        self.num_agents = env_config["num_agents"]
        self.env = env
        self.action_space = action_space
        self.ori_obs_space = obs_space
        self.observation_space = GymDict({"obs": Box(
            low=-100.0,
            high=100.0,
            shape=(obs_dim,),
            dtype=np.float32)})
        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]

        # back to be compatible in run script
        env_config["map_name"] = map_name

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for index in range(self.num_agents):
            obs_one_agent = []
            for key in original_obs.keys():
                if key in self.order_obs_key:
                    obs_one_agent.append(np.squeeze(np.array(original_obs[key][index], dtype=np.float32)))
            obs["agent_{}".format(index)] = {
                "obs": np.hstack(obs_one_agent)
            }
        return obs

    def step(self, action_dict):
        actions = {}
        action_start = 0
        for action_name in self.ori_action_space.keys():
            action_ls = []
            if len(self.ori_action_space[action_name][0].shape) > 0:
                action_dim = self.ori_action_space[action_name][0].shape[0]
                for agent_name in self.agents:
                    agent_action = action_dict[agent_name]
                    action_value = agent_action[action_start: action_start + action_dim]
                    action_ls.append(action_value)
            else:
                action_dim = 1
                for agent_name in self.agents:
                    agent_action = action_dict[agent_name]
                    action_value = agent_action[action_start: action_start + action_dim][0]
                    action_ls.append(action_value)

            action_start += action_dim
            actions[action_name] = np.array(action_ls)

        o, r, d, i = self.env.step(actions)
        rewards = {}
        obs = {}
        for agent_index, agent_name in enumerate(self.agents):
            rewards[agent_name] = r[agent_index]
            obs_one_agent = []
            for key in o.keys():
                if key in self.order_obs_key:
                    obs_one_agent.append(np.squeeze(np.array(o[key][agent_index], dtype=np.float32)))
            obs[agent_name] = {
                "obs": np.hstack(obs_one_agent)
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
            "episode_limit": self.env_config["env_horizon"],
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
