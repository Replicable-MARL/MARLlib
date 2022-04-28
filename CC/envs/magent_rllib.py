from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Box
import supersuit as ss
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from pettingzoo.magent import adversarial_pursuit_v3, battle_v3, battlefield_v3, combined_arms_v5, gather_v3, \
    tiger_deer_v3

REGISTRY = {}
REGISTRY["adversarial_pursuit"] = adversarial_pursuit_v3.parallel_env
REGISTRY["battle"] = battle_v3.parallel_env
REGISTRY["battlefield"] = battlefield_v3.parallel_env
REGISTRY["combined_arms"] = combined_arms_v5.parallel_env
REGISTRY["gather"] = gather_v3.parallel_env
REGISTRY["tiger_deer"] = tiger_deer_v3.parallel_env

mini_channel_dim_dict = {
    "adversarial_pursuit": 4,
    "battle": 4,
    "battlefield": 4,
    "combined_arms": 6,
    "gather": 4,
    "tiger_deer": 6,
}


class RllibMAgent(MultiAgentEnv):

    def __init__(self, env_config):
        map = env_config["map_name"]
        env_config.pop("map_name", None)
        env = REGISTRY[map](**env_config)

        # keep obs and action dim same across agents
        # pad_action_space_v0 will auto mask the padding actions
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)

        self.env = ParallelPettingZooEnv(env)
        self.mini_channel_dim = mini_channel_dim_dict[map]
        self.action_space = self.env.action_space
        self.observation_space = GymDict({
            "obs": Box(
                low=self.env.observation_space.low[:, :, :-self.mini_channel_dim],
                high=self.env.observation_space.high[:, :, :-self.mini_channel_dim],
                dtype=self.env.observation_space.dtype),
            "state": Box(
                low=self.env.observation_space.low[:, :, -self.mini_channel_dim:],
                high=self.env.observation_space.high[:, :, -self.mini_channel_dim:],
                dtype=self.env.observation_space.dtype),
        })
        self.agents = self.env.agents
        self.num_agents = len(self.agents)
        env_config["map_name"] = map
        # env_config["mini_channel_dim"] = self.mini_channel_dim
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for key in self.agents:
            obs[key] = {
                "obs": original_obs[key][:, :, :-self.mini_channel_dim],
                "state": original_obs[key][:, :, -self.mini_channel_dim:]
            }
        return obs

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        rewards = {}
        obs = {}
        for key in action_dict.keys():
            rewards[key] = r[key]
            obs[key] = {
                "obs": o[key][:, :, :-self.mini_channel_dim],
                "state": o[key][:, :, -self.mini_channel_dim:]
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
