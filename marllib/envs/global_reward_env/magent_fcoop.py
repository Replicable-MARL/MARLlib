# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from marllib.envs.base_env.magent import RLlibMAgent

legal_scenarios = ["gather"]


class RLlibMAgent_FCOOP(RLlibMAgent):

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

