from rware import Warehouse, RewardType
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict, Discrete, Box
import supersuit as ss
from pettingzoo.mpe import simple_adversary_v2, simple_crypto_v2, simple_v2, simple_push_v2, simple_tag_v2, \
    simple_spread_v2, simple_reference_v2, simple_world_comm_v2, simple_speaker_listener_v3
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv

_sizes = {
    "tiny": (1, 3),
    "small": (2, 3),
    "medium": (2, 5),
    "large": (3, 5),
}
_difficulty = {"easy": 2, "medium": 1, "hard": 0.5}


class RllibMPE_QMIX(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def __init__(self, env):

        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
        self.env = ParallelPettingZooEnv(env)

        self.action_space = self.env.action_space
        self.observation_space = Dict({"obs": Box(
            low=-100.0,
            high=100.0,
            shape=(self.env.observation_space.shape[0],),
            dtype=self.env.observation_space.dtype)})
        self.agents = self.env.agents
        self.num_agents = len(self.agents)

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for i in self.agents:
            obs[i] = {"obs": original_obs[i]}
        return obs

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        rewards = {}
        obs = {}
        for key in action_dict.keys():
            rewards[key] = r[key]
            obs[key] = {
                "obs": o[key]
            }
        dones = {"__all__": d["__all__"]}
        return obs, rewards, dones, info

    def close(self):
        self.env.close()


# class RllibMPE_QMIX(MultiAgentEnv):
#     """An example of a wrapper for GFootball to make it compatible with rllib."""
#
#     def __init__(self, env):
#
#         env = ss.pad_observations_v0(env)
#         env = ss.pad_action_space_v0(env)
#         self.env = PettingZooEnv(env)
#
#         self.action_space = self.env.action_space
#         self.observation_space = Dict({"obs": Box(
#             low=-100.0,
#             high=100.0,
#             shape=(self.env.observation_space.shape[0],),
#             dtype=self.env.observation_space.dtype)})
#         self.agents = self.env.agents
#         self.num_agents = len(self.agents)
#
#         self.t = 0
#
#     def reset(self):
#         original_obs = self.env.reset()
#         obs = {}
#         for key in original_obs.keys():
#             obs[key] = {"obs": original_obs[key]}
#         return obs
#
#     def step(self, action_dict):
#         o, r, d, i = self.env.step(action_dict)
#         for key in o.keys():
#             o[key] = {"obs": o[key]}
#         dones = {"__all__": d["__all__"]}
#         return o, r, dones, i