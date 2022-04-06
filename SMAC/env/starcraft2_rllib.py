from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from smac.env.starcraft2.starcraft2 import StarCraft2Env
import numpy as np
from absl import logging
from pysc2.lib import protocol
from s2clientprotocol import sc2api_pb2 as sc_pb
from gym.spaces import Dict, Discrete, Box


class StarCraft2Env_Rllib(MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """

    def __init__(self, map_name):
        map_name = map_name if isinstance(map_name, str) else map_name["map_name"]
        self.env = StarCraft2Env(map_name)

        env_info = self.env.get_env_info()
        self.n_agents = self.env.n_agents
        self.n_actions = self.env.n_actions

        obs_shape = env_info['obs_shape']
        n_actions = env_info['n_actions']
        state_shape = env_info['state_shape']
        self.observation_space = Dict({
                                  "obs": Box(-2.0, 2.0, shape=(obs_shape,)),
                                  "state": Box(-2.0, 2.0, shape=(state_shape,)),
                                  "action_mask": Box(-2.0, 2.0, shape=(n_actions,))
                              })
        self.action_space = Discrete(n_actions)

    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self.env.reset()
        obs_smac_api = self.env.get_obs()
        state_smac_api = self.env.get_state()
        obs_rllib = {}
        for agent_index in range(self.n_agents):
            obs_one_agent = obs_smac_api[agent_index]
            state_one_agent = state_smac_api
            action_mask_one_agent = np.array(self.env.get_avail_agent_actions(agent_index)).astype(np.float32)
            agent_index = "agent_{}".format(agent_index)
            obs_rllib[agent_index] = {
                "obs": obs_one_agent,
                "state": state_one_agent,
                "action_mask": action_mask_one_agent
            }

        return obs_rllib

    def step(self, actions):

        actions_ls = [int(actions[agent_id]) for agent_id in actions.keys()]

        reward, terminated, info = self.env.step(actions_ls)

        obs_smac_api = self.env.get_obs()
        state_smac_api = self.env.get_state()
        obs_rllib = {}
        reward_rllib = {}
        for agent_index in range(self.n_agents):
            obs_one_agent = obs_smac_api[agent_index]
            state_one_agent = state_smac_api
            action_mask_one_agent = np.array(self.env.get_avail_agent_actions(agent_index)).astype(np.float32)
            agent_index = "agent_{}".format(agent_index)
            obs_rllib[agent_index] = {
                "obs": obs_one_agent,
                "state": state_one_agent,
                "action_mask": action_mask_one_agent
            }
            reward_rllib[agent_index] = reward

        dones = {"__all__": terminated}

        return obs_rllib, reward_rllib, dones, {}

    def get_env_info(self):
        return self.env.get_env_info()
