import numpy as np
from marllib.envs.base_env.aircombat import RLlibCloseAirCombatEnv


class RLlibCloseAirCombatEnv_FCOOP(RLlibCloseAirCombatEnv):

    def __init__(self, env_config):
        super().__init__(env_config)
        if "vsBaseline" not in self.env_args["map_name"]:
            raise ValueError(self.env_args["map_name"] + "is not in cooperative mode")

    def reset(self):
        original_obs, _ = self.env.reset()
        obs = {}
        for index, agent in enumerate(self.agents):
            obs[agent] = {
                "obs": np.float32(original_obs[index])
            }
        return obs

    def step(self, action_dict):
        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)
        o, _, r, d, i = self.env.step(np.array(actions))
        rewards = {}
        obs = {}
        for index, agent in enumerate(self.agents):
            rewards[agent] = r[index][0]  # rewards are shared in default
            obs[agent] = {
                "obs": np.float32(o[index])
            }
        done = {"__all__": True if d.sum() == self.num_agents else False}
        return obs, rewards, done, {}
