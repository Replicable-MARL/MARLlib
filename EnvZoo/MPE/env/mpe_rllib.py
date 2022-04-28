from rware import Warehouse, RewardType
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict, Discrete, Box
import supersuit as ss
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv


class RllibMPE(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def __init__(self, env):
        # keep obs and action dim same across agents
        # pad_action_space_v0 will auto mask the padding actions
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)

        self.env = PettingZooEnv(env)

        self.action_space = self.env.action_space
        self.observation_space = Box(
            low=-100.0,
            high=100.0,
            shape=(self.env.observation_space.shape[0],),
            dtype=self.env.observation_space.dtype)
        self.agents = self.env.agents
        self.num_agents = len(self.agents)

    def reset(self):
        original_obs = self.env.reset()
        return original_obs

    def step(self, action_dict):
        o, r, d, i = self.env.step(action_dict)
        dones = {"__all__": d["__all__"]}
        return o, r, dones, i
