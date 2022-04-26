import os
import tempfile

import argparse
import gfootball.env as football_env
import gym
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from gym.spaces import Dict, Discrete, Box

"""
currently, rllib built-in qmix algo needs the returned observation in step() function as a dict, 
containing dict {obs: original observation}
"""


class RllibGFootball_QMIX(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def __init__(self, env_config):
        self.env = football_env.create_environment(
            env_name=env_config["env_name"],
            logdir=os.path.join(tempfile.gettempdir(), "rllib"),
            render=False,
            dump_frequency=0,
            number_of_left_players_agent_controls=env_config["num_agents"],
            channel_dimensions=(42, 42))
        self.action_space = Discrete(self.env.action_space.nvec[1])
        self.observation_space = Dict({"obs": Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            dtype=self.env.observation_space.dtype)})
        # self.observation_space = gym.spaces.Box(
        #     low=self.env.observation_space.low[0],
        #     high=self.env.observation_space.high[0],
        #     dtype=self.env.observation_space.dtype)
        self.num_agents = env_config["num_agents"]

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for x in range(self.num_agents):
            if self.num_agents > 1:
                obs["agent_%d" % x] = {
                    "obs": original_obs[x]
                }
            else:
                obs["agent_%d" % x] = {
                    original_obs
                }
        return obs

    def step(self, action_dict):
        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)
        o, r, d, i = self.env.step(actions)
        rewards = {}
        obs = {}
        infos = {}
        for pos, key in enumerate(sorted(action_dict.keys())):
            infos[key] = i
            if self.num_agents > 1:
                rewards[key] = r[pos]
                obs[key] = {
                    "obs": o[pos]
                }
            else:
                rewards[key] = r
                obs[key] = {
                    "obs": o
                }
        dones = {"__all__": d}
        return obs, rewards, dones, infos
