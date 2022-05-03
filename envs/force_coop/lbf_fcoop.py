from envs.base_env.lbf import RllibLBF


class RllibLBF_FCOOP(RllibLBF):

    def __init__(self, env_config):
        env_config["force_coop"] = True
        super().__init__(env_config)

    def step(self, action_dict):
        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)
        o, r, d, i = self.env.step(tuple(actions))
        r = sum(r)
        rewards = {}
        obs = {}
        infos = {}
        done_flag = False
        for pos, key in enumerate(sorted(action_dict.keys())):
            infos[key] = i
            rewards[key] = r
            obs[key] = {
                "obs": o[pos]
            }
            done_flag = d[pos] or done_flag
        dones = {"__all__": done_flag}
        return obs, rewards, dones, infos
