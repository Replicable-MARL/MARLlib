from envs.base_env.football import RllibGFootball


class RllibGFootball_FCOOP(RllibGFootball):

    def step(self, action_dict):
        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)
        o, r, d, i = self.env.step(actions)
        r = sum(r)
        rewards = {}
        obs = {}
        infos = {}
        for pos, key in enumerate(sorted(action_dict.keys())):
            infos[key] = i
            rewards[key] = r
            obs[key] = {
                "obs": o[pos]
            }
        dones = {"__all__": d}
        return obs, rewards, dones, infos
