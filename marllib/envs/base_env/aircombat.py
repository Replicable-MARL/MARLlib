import numpy as np

from marllib.patch.aircombat.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
import torch
import gym
from gym.spaces import Dict as GymDict, Box
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.utils import merge_dicts

policy_mapping_dict = {
    "MultipleCombat_2v2/NoWeapon/Selfplay": {
        "description": "aircombat AI vs AI",
        "team_prefix": ("teamA_", "teamB_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": True,
    },
    "MultipleCombat_2v2/NoWeapon/vsBaseline": {
        "description": "aircombat AI vs Bot",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    "MultipleCombat_4v4/NoWeapon/vsBaseline": {
        "description": "aircombat AI vs Bot",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


class RLlibCloseAirCombatEnv(MultiAgentEnv):

    def __init__(self, env_config):
        self.env_args = env_config
        self.env = self.get_env(env_config)
        self.num_agents = self.env.num_agents
        self.observation_space = GymDict({"obs": self.env.observation_space})
        self.action_space = self.env.action_space
        self.episode_limit = self.env.config.max_steps
        if "vsBaseline" in env_config["map_name"]:
            self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        else:  # AI vs AI
            agent_dict = self.env.agents
            self.teamA_agent_num = sum(1 if "A" in agent_name else 0 for agent_name in agent_dict.keys())
            self.agents_teamA = ["teamA_{}".format(i) for i in range(self.teamA_agent_num)]
            self.teamB_agent_num = sum(1 if "B" in agent_name else 0 for agent_name in agent_dict.keys())
            self.agents_teamB = ["teamB_{}".format(i) for i in range(self.teamB_agent_num)]
            self.agents = self.agents_teamA + self.agents_teamB

    def reset(self):
        original_obs, _ = self.env.reset()
        obs = {}
        if "vsBaseline" in self.env_args["map_name"]:
            for index, agent in enumerate(self.agents):
                obs[agent] = {
                    "obs": np.float32(original_obs[index])
                }
        else:
            for index, agent in enumerate(self.agents_teamA):
                obs[agent] = {
                    "obs": np.float32(original_obs[index])
                }
            for index, agent in enumerate(self.agents_teamB):
                obs[agent] = {
                    "obs": np.float32(original_obs[index + self.teamA_agent_num])
                }
        return obs

    def step(self, action_dict):
        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)
        o, _, r, d, i = self.env.step(np.array(actions))
        rewards = {}
        obs = {}
        if "vsBaseline" in self.env_args["map_name"]:
            for index, agent in enumerate(self.agents):
                rewards[agent] = r[index][0]
                obs[agent] = {
                    "obs": np.float32(o[index])
                }
        else:
            for index, agent in enumerate(self.agents_teamA):
                rewards[agent] = r[index][0]
                obs[agent] = {
                    "obs": np.float32(o[index])
                }
            for index, agent in enumerate(self.agents_teamB):
                rewards[agent] = r[index + self.teamA_agent_num][0]
                obs[agent] = {
                    "obs": np.float32(o[index + self.teamA_agent_num])
                }
        done = {"__all__": True if d.sum() == self.num_agents else False}
        return obs, rewards, done, {}

    def close(self):
        self.env.close()

    def get_env(self, env_args):
        task = env_args["map_name"].split("_")[0]
        scenario = env_args["map_name"].split("_")[1]
        if task in ["SingleCombat", "SingleControl"]:
            print()
            raise ValueError("Can not support the " +
                             task + "environment." +
                             "\nMARLlib is built for multi-agent settings")
        elif task == "MultipleCombat":
            env = MultipleCombatEnv(scenario)
        else:
            raise NotImplementedError("Can not support the " +
                                      task + "environment.")
        return env

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.episode_limit,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
