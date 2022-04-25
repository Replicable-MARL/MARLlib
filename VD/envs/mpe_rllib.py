from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict, Discrete, Box
import supersuit as ss
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from pettingzoo.mpe import simple_spread_v2, simple_reference_v2, simple_speaker_listener_v3

REGISTRY = {}
REGISTRY["simple_spread"] = simple_spread_v2.parallel_env
REGISTRY["simple_reference"] = simple_reference_v2.parallel_env
REGISTRY["simple_speaker_listener"] = simple_speaker_listener_v3.parallel_env


class RllibMPE(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def __init__(self, env_config):
        map = env_config["map_name"]
        env_config.pop("map_name", None)
        self.env_config = env_config
        env = REGISTRY[map](**self.env_config)

        # keep obs and action dim same across agents
        # pad_action_space_v0 will auto mask the padding actions
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
        env_config["map_name"] = map

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for i in self.agents:
            obs[i] = {"obs": original_obs[i]}
        return obs

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        # cooperative need global reward (specific to football)
        reward = 0
        for key in r.keys():
            reward += r[key]
        rewards = {}
        obs = {}
        for key in action_dict.keys():
            rewards[key] = reward
            obs[key] = {
                "obs": o[key]
            }
        dones = {"__all__": d["__all__"]}
        return obs, rewards, dones, info

    def close(self):
        self.env.close()

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 200
        }
        return env_info
