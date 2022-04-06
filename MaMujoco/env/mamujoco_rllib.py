import os
import tempfile

import argparse
# import gfootball.env as football_env
import gym
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from MaMujoco.src.multiagent_mujoco.mujoco_multi import MujocoMulti
from gym.spaces import Dict, Discrete, Box
import numpy as np

class RllibMAMujoco(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def __init__(self, env_config):
        self.env = MujocoMulti(env_args=env_config)
        self.action_space = self.env.action_space[0]
        self.state_dim = self.env.wrapped_env.observation_space.shape[0]
        # self.observation_space = gym.spaces.Box(
        #     low=-100.0,
        #     high=100.0,
        #     shape=(self.env.obs_size,),
        #     dtype=self.env.observation_space[0].dtype)
        self.observation_space = Dict({
            "obs": Box(-100.0, 100.0, shape=(self.env.obs_size,), dtype=self.env.observation_space[0].dtype),
            "state": Box(-100.0, 100.0, shape=(self.state_dim,), dtype=self.env.observation_space[0].dtype),
        })

        if "|" in env_config["agent_conf"]:
            self.num_agents = len(env_config["agent_conf"].split("|"))
        else:
            self.num_agents = int(env_config["agent_conf"].split("x")[0])

        self.previous_r = None

    def reset(self):
        self.env.reset()
        o = self.env.get_obs()  # obs
        s = self.env.get_state()  # g state
        # to float32 for RLLIB check
        obs = {}
        for x in range(self.num_agents):
            obs["agent_%d" % x] = {
                "obs": np.float32(o[x]),
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


def normalize_action(action, action_space):
    action = (action + 1) / 2
    action *= (action_space.high - action_space.low)
    action += action_space.low
    return action
