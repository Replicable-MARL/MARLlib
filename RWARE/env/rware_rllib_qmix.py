from rware import Warehouse, RewardType
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict, Discrete, Box

_sizes = {
    "tiny": (1, 3),
    "small": (2, 3),
    "medium": (2, 5),
    "large": (3, 5),
}
_difficulty = {"easy": 2, "medium": 1, "hard": 0.5}


class RllibRWARE_QMIX(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def __init__(self, env_config):
        self.env = Warehouse(
            shelf_rows=_sizes[env_config["map_size"]][0],
            shelf_columns=_sizes[env_config["map_size"]][1],
            n_agents=env_config["agents_num"],
            request_queue_size=int(env_config["agents_num"] * _difficulty[env_config["difficulty"]]),

            # default setting
            column_height=8,
            max_inactivity_steps=None,
            max_steps=500,
            reward_type=RewardType.INDIVIDUAL,
            fast_obs=True,
            msg_bits=0,
            sensor_range=1,
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