from envs.base_env.rware import RllibRWARE


class RllibRWARE_FCOOP(RllibRWARE):

    def step(self, action_dict):
        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)
        o, r, d, i = self.env.step(actions)
        # cooperative need global reward
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
