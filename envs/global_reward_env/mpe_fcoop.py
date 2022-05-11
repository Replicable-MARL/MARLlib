from envs.base_env.mpe import RllibMPE

legal_scenarios = ["simple_spread", "simple_reference", "simple_speaker_listener"]


class RllibMPE_FCOOP(RllibMPE):

    def __init__(self, env_config):
        if env_config["map_name"] not in legal_scenarios:
            raise ValueError("must in: 1.simple_spread, 2.simple_reference, 3.simple_speaker_listener")
        super().__init__(env_config)

    def step(self, action_dict):
        o, r, d, info = self.env.step(action_dict)
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
