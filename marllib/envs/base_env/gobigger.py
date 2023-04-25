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

import copy

from gobigger.envs import create_env_custom
from gym.spaces import Dict as GymDict, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np


policy_mapping_dict = {
    "all_scenario": {
        "description": "mixed scenarios to t>2 (num_teams > 1)",
        "team_prefix": ("team0_", "team1_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


class RLlibGoBigger(MultiAgentEnv):

    def __init__(self, env_config):

        map_name = env_config["map_name"]

        env_config.pop("map_name", None)
        self.num_agents_per_team = int(map_name.split("p")[-1][0])
        self.num_teams = int(map_name.split("_t")[1][0])
        if self.num_teams == 1:
            policy_mapping_dict["all_scenario"]["team_prefix"] = ("team0_",)
        self.num_agents = self.num_agents_per_team * self.num_teams
        self.max_steps = env_config["frame_limit"]
        self.env = create_env_custom(type='st', cfg=dict(
            team_num=self.num_teams,
            player_num_per_team=self.num_agents_per_team,
            frame_limit=self.max_steps
        ))

        self.action_space = Box(low=-1,
                                high=1,
                                shape=(2,),
                                dtype=float)

        self.rectangle_dim = 4
        self.food_dim = self.num_agents * 100
        self.thorns_dim = self.num_agents * 6
        self.clone_dim = self.num_agents * 10
        self.team_name_dim = 1
        self.score_dim = 1

        self.obs_dim = self.rectangle_dim + self.food_dim + self.thorns_dim + \
                       self.clone_dim + self.team_name_dim + self.score_dim

        self.observation_space = GymDict({"obs": Box(
            low=-1e6,
            high=1e6,
            shape=(self.obs_dim,),
            dtype=float)})

        self.agents = []
        for team_index in range(self.num_teams):
            for agent_index in range(self.num_agents_per_team):
                self.agents.append("team{}_{}".format(team_index, agent_index))

        env_config["map_name"] = map_name
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for agent_index, agent_name in enumerate(self.agents):

            rectangle = list(original_obs[1][agent_index]["rectangle"])

            overlap_dict = original_obs[1][agent_index]["overlap"]

            food = overlap_dict["food"]
            if 4 * len(food) > self.food_dim:
                food = food[:self.food_dim // 4]
            else:
                padding = [0] * (self.food_dim - 4 * len(food))
                food.append(padding)
            food = [item for sublist in food for item in sublist]

            thorns = overlap_dict["thorns"]
            if 6 * len(thorns) > self.thorns_dim:
                thorns = thorns[:self.thorns_dim // 6]
            else:
                padding = [0] * (self.thorns_dim - 6 * len(thorns))
                thorns.append(padding)
            thorns = [item for sublist in thorns for item in sublist]

            clone = overlap_dict["clone"]
            if 10 * len(clone) > self.clone_dim:
                clone = clone[:self.clone_dim // 10]
            else:
                padding = [0] * (self.clone_dim - 10 * len(clone))
                clone.append(padding)
            clone = [item for sublist in clone for item in sublist]

            team = original_obs[1][agent_index]["team_name"]
            score = original_obs[1][agent_index]["score"]

            all_elements = rectangle + food + thorns + clone + [team] + [score]
            all_elements = np.array(all_elements, dtype=float)

            obs[agent_name] = {
                "obs": all_elements
            }

        return obs

    def step(self, action_dict):
        actions = {}
        for i, agent_name in enumerate(self.agents):
            actions[i] = list(action_dict[agent_name])
            actions[i].append(-1)

        original_obs, team_rewards, done, info = self.env.step(actions)

        rewards = {}
        obs = {}
        infos = {}

        for agent_index, agent_name in enumerate(self.agents):

            rectangle = list(original_obs[1][agent_index]["rectangle"])

            overlap_dict = original_obs[1][agent_index]["overlap"]

            food = overlap_dict["food"]
            if 4 * len(food) > self.food_dim:
                food = food[:self.food_dim // 4]
            else:
                padding = [0] * (self.food_dim - 4 * len(food))
                food.append(padding)
            food = [item for sublist in food for item in sublist]

            thorns = overlap_dict["thorns"]
            if 6 * len(thorns) > self.thorns_dim:
                thorns = thorns[:self.thorns_dim // 6]
            else:
                padding = [0] * (self.thorns_dim - 6 * len(thorns))
                thorns.append(padding)
            thorns = [item for sublist in thorns for item in sublist]

            clone = overlap_dict["clone"]
            if 10 * len(clone) > self.clone_dim:
                clone = clone[:self.clone_dim // 10]
            else:
                padding = [0] * (self.clone_dim - 10 * len(clone))
                clone.append(padding)
            clone = [item for sublist in clone for item in sublist]

            team = original_obs[1][agent_index]["team_name"]
            score = original_obs[1][agent_index]["score"]

            all_elements = rectangle + food + thorns + clone + [team] + [score]
            all_elements = np.array(all_elements, dtype=float)

            obs[agent_name] = {
                "obs": all_elements
            }

            rewards[agent_name] = team_rewards[team]

        dones = {"__all__": done}
        return obs, rewards, dones, infos

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.max_steps,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info

    def close(self):
        self.env.close()
