from gym_derk.envs import DerkEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict, Discrete, Box
import random


class RllibDerk(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def __init__(self, env_config):
        self.env = DerkEnv(
            n_arenas=1,
            turbo_mode=True,
            agent_server_args={"port": random.randint(8888, 9999)}
        )
        self.action_space = self.env.action_space
        # self.observation_space = self.env.observation_space
        self.observation_space = Box(
            low=-100.0,
            high=100.0,
            shape=(self.env.observation_space.shape[0],),
            dtype=self.env.observation_space.dtype)
        self.num_agents = self.env.n_agents

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for x in range(self.num_agents):
            if self.num_agents > 1:
                obs["agent_%d" % x] = original_obs[x]
            else:
                obs["agent_%d" % x] = original_obs
        return obs

    def step(self, action_dict):
        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)
        o, r, d, i = self.env.step(tuple(actions))
        rewards = {}
        obs = {}
        infos = {}
        done_flag = False
        for pos, key in enumerate(sorted(action_dict.keys())):
            infos[key] = i
            if self.num_agents > 1:
                rewards[key] = r[pos]
                obs[key] = o[pos]
            else:
                rewards[key] = r
                obs[key] = o
        dones = {"__all__": d[0]}
        return obs, rewards, dones, infos
