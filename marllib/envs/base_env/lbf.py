from lbforaging.foraging import ForagingEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Discrete, Box

policy_mapping_dict = {
    "all_scenario": {
        "description": "lbf all scenarios",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}

class RllibLBF(MultiAgentEnv):

    def __init__(self, env_config):
        map_name = env_config["map_name"]
        env_config.pop("map_name", None)
        field_size_y = env_config.pop("field_size_y", None)
        field_size_x = env_config.pop("field_size_x", None)

        env_config["field_size"] = (field_size_y, field_size_x)
        self.env = ForagingEnv(**env_config)

        self.action_space = self.env.action_space[0]
        self.observation_space = GymDict({"obs": Box(
            low=-100.0,
            high=100.0,
            shape=(self.env.observation_space[0].shape[0],),
            dtype=self.env.observation_space[0].dtype)})
        self.num_agents = self.env.n_agents
        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        env_config["field_size_y"] = field_size_y
        env_config["field_size_x"] = field_size_x
        env_config["map_name"] = map_name
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for x in range(self.num_agents):
            obs["agent_%d" % x] = {
                "obs": original_obs[x]
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
            rewards[key] = r[pos]
            obs[key] = {
                "obs": o[pos]
            }
            done_flag = d[pos] or done_flag
        dones = {"__all__": done_flag}
        return obs, rewards, dones, infos

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env_config["max_episode_steps"],
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info

    def close(self):
        self.env.close()
