from marllib.envs.base_env.magent import RllibMAgent

legal_scenarios = ["gather"]


class RllibMAgent_FCOOP(RllibMAgent):

    def __init__(self, env_config):
        if env_config["map_name"] not in legal_scenarios:
            raise ValueError("must in: 1.gather")
        super().__init__(env_config)

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for key in original_obs.keys():
            obs[key] = {
                "obs": original_obs[key][:, :, :-self.mini_channel_dim],
                "state": original_obs[key][:, :, -self.mini_channel_dim:]
            }
        return obs

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
        reward = 0
        for key in r.keys():
            reward += r[key]
        rewards = {}
        obs = {}
        for key in o.keys():
            rewards[key] = reward/self.num_agents
            obs[key] = {
                "obs": o[key][:, :, :-self.mini_channel_dim],
                "state": o[key][:, :, -self.mini_channel_dim:]
            }
        dones = {"__all__": d["__all__"]}
        return obs, rewards, dones, info

