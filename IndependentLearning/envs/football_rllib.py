import gfootball.env as football_env
import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.utils import merge_dicts

SMM_WIDTH = 42
SMM_HEIGHT = 42

ally_num_dict = {
    "academy_empty_goal": 2,
    "academy_empty_goal_close": 2,
    "academy_run_to_score_with_keeper": 2,
    "academy_run_to_score": 2,
    "academy_pass_and_shoot_with_keeper": 3,
    "academy_run_pass_and_shoot_with_keeper": 3,
    "academy_3_vs_1_with_keeper": 4,
    "academy_single_goal_versus_lazy": 11,
    "academy_corner": 11,
    "academy_counterattack_easy": 11,
    "academy_counterattack_hard": 11,
}


class RllibGFootball(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def __init__(self, env_config):
        env_config["env_name"] = env_config.pop("map_name")
        self.env_config = env_config

        extra_setting = {
            "number_of_left_players_agent_controls": ally_num_dict[self.env_config["env_name"]],
            "channel_dimensions": (SMM_WIDTH, SMM_HEIGHT),
        }

        self.env = football_env.create_environment(**merge_dicts(self.env_config, extra_setting))
        self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            dtype=self.env.observation_space.dtype)
        self.num_agents = ally_num_dict[self.env_config["env_name"]]

        # back to be compatible in run script
        env_config["map_name"] = env_config.pop("env_name")

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
        o, r, d, i = self.env.step(actions)
        rewards = {}
        obs = {}
        infos = {}
        for pos, key in enumerate(sorted(action_dict.keys())):
            infos[key] = i
            if self.num_agents > 1:
                rewards[key] = r[pos]
                obs[key] = o[pos]
            else:
                rewards[key] = r
                obs[key] = o
        dones = {"__all__": d}
        return obs, rewards, dones, infos

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
