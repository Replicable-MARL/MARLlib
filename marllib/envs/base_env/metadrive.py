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
from metadrive.envs.marl_envs import MultiAgentBottleneckEnv, MultiAgentParkingLotEnv, MultiAgentRoundaboutEnv, \
    MultiAgentIntersectionEnv, MultiAgentTollgateEnv
from collections import defaultdict
from metadrive.utils import norm
import numpy as np

SUPER_REGISTRY = {}
SUPER_REGISTRY["Bottleneck"] = MultiAgentBottleneckEnv
SUPER_REGISTRY["ParkingLot"] = MultiAgentParkingLotEnv
SUPER_REGISTRY["Intersection"] = MultiAgentIntersectionEnv
SUPER_REGISTRY["Roundabout"] = MultiAgentRoundaboutEnv
SUPER_REGISTRY["Tollgate"] = MultiAgentTollgateEnv
NE_distance = 10

policy_mapping_dict = {
    "all_scenario": {
        "description": "metadrive all scenarios",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


def dynamic_inheritance(super_class):
    class RLlibMetaDrive_Scenario(super_class):

        def __init__(self, config):
            map = config["map_name"]
            config.pop("map_name", None)
            super(super_class, self).__init__(config)
            self.__name__ = map
            self.__qualname__ = map
            self.neighbours_distance = NE_distance
            self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))

        def step(self, actions):
            obs, reward, done, info = super(super_class, self).step(actions)
            update_neighbours_map(self.distance_map, self.vehicles, reward, info, self.config)
            return obs, reward, done, info

    return RLlibMetaDrive_Scenario


class RLlibMetaDrive(MultiAgentEnv):

    def __init__(self, env_config):
        map = env_config["map_name"]
        super_class = SUPER_REGISTRY[map]
        env_class = dynamic_inheritance(super_class)
        self.env = env_class(env_config)

        self.action_space = self.env.action_space["agent0"]
        self.observation_space = GymDict({
            "obs": Box(
                low=self.env.observation_space["agent0"].low,
                high=self.env.observation_space["agent0"].high,
                dtype=self.env.observation_space["agent0"].dtype),
        })

        self.num_agents = self.env.num_agents
        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        env_config["map_name"] = map
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for key in original_obs.keys():
            new_key = "agent_{}".format(key[5:])
            obs[new_key] = {"obs": np.float32(original_obs[key])}
        return obs

    def step(self, action_dict):
        # reset the key:
        new_action_dict = {}
        for key in action_dict.keys():
            new_action_dict["agent{}".format(key[6:])] = action_dict[key]
        o, r, d, info = self.env.step(new_action_dict)
        rewards = {}
        obs = {}
        for key in o.keys():
            new_key = "agent_{}".format(key[5:])
            rewards[new_key] = r[key]
            obs[new_key] = {
                "obs": np.float32(o[key])
            }
        dones = {"__all__": d["__all__"]}
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


def update_neighbours_map(distance_map, vehicles, reward, info, config):
    distance_map.clear()
    keys = list(vehicles.keys())
    for c1 in range(0, len(keys) - 1):
        for c2 in range(c1 + 1, len(keys)):
            k1 = keys[c1]
            k2 = keys[c2]
            p1 = vehicles[k1].position
            p2 = vehicles[k2].position
            distance = norm(p1[0] - p2[0], p1[1] - p2[1])
            distance_map[k1][k2] = distance
            distance_map[k2][k1] = distance

    for kkk in info.keys():
        neighbours, nei_distances = find_in_range(kkk, config["neighbours_distance"], distance_map)
        info[kkk]["neighbours"] = neighbours
        info[kkk]["neighbours_distance"] = nei_distances
        nei_rewards = [reward[kkkkk] for kkkkk in neighbours]
        if nei_rewards:
            info[kkk]["nei_rewards"] = sum(nei_rewards) / len(nei_rewards)
        else:
            # i[kkk]["nei_rewards"] = r[kkk]
            info[kkk]["nei_rewards"] = 0.0  # Do not provides neighbour rewards
        info[kkk]["global_rewards"] = sum(reward.values()) / len(reward.values())


def find_in_range(v_id, distance, distance_map):
    if distance <= 0:
        return []
    max_distance = distance
    dist_to_others = distance_map[v_id]
    dist_to_others_list = sorted(dist_to_others, key=lambda k: dist_to_others[k])
    ret = [
        dist_to_others_list[i] for i in range(len(dist_to_others_list))
        if dist_to_others[dist_to_others_list[i]] < max_distance
    ]
    ret2 = [
        dist_to_others[dist_to_others_list[i]] for i in range(len(dist_to_others_list))
        if dist_to_others[dist_to_others_list[i]] < max_distance
    ]
    return ret, ret2
