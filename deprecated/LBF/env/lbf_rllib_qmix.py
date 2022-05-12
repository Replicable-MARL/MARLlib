from lbforaging.foraging import ForagingEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict, Discrete, Box


class RllibLBF_QMIX(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def __init__(self, env_config):
        self.env = ForagingEnv(
            players=env_config["num_agents"],
            field_size=(env_config["field_size"], env_config["field_size"]),
            max_food=env_config["max_food"],
            sight=env_config["sight"],
            force_coop=env_config["force_coop"],  # True or False
            # default setting
            max_episode_steps=env_config["max_episode_steps"],
            max_player_level=3,
        )
        self.action_space = self.env.action_space[0]
        self.observation_space = Dict({"obs": Box(
            low=-100.0,
            high=100.0,
            shape=(self.env.observation_space[0].shape[0],),
            dtype=self.env.observation_space[0].dtype)})
        self.num_agents = self.env.n_agents

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
                    "obs": original_obs
                }
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
                obs[key] = {
                    "obs": o[pos]
                }
            else:
                rewards[key] = r
                obs[key] = {
                    "obs": o
                }
            done_flag = d[pos] or done_flag
        dones = {"__all__": done_flag}
        return obs, rewards, dones, infos
