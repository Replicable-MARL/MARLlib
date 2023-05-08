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

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from marllib.patch.dpn.var_voltage_control.voltage_control_env import VoltageControl
import numpy as np
from gym.spaces import Dict as GymDict, Box
import os

policy_mapping_dict = {
    "all_scenario": {
        "description": "voltage control all scenarios",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}

global_data_source_path = os.getcwd()
max_steps = 1000


class RLlibVoltageControl(MultiAgentEnv):

    def __init__(self, env_config):
        net_topology = env_config.pop("map_name", None)
        # net_topology = env_config.pop("net_topology", None)
        data_path = env_config["data_path"].split("/")
        # net_topology = "case322_3min_final"  # case33_3min_final / case141_3min_final / case322_3min_final
        data_path[-1] = net_topology
        env_config["data_path"] = "/".join(data_path)

        # set the action range
        assert net_topology in ['case33_3min_final', 'case141_3min_final',
                                'case322_3min_final'], f'{net_topology} is not a valid scenario.'
        if net_topology == 'case33_3min_final':
            env_config["action_bias"] = 0.0
            env_config["action_scale"] = 0.8
        elif net_topology == 'case141_3min_final':
            env_config["action_bias"] = 0.0
            env_config["action_scale"] = 0.6
        elif net_topology == 'case322_3min_final':
            env_config["action_bias"] = 0.0
            env_config["action_scale"] = 0.8

        # define control mode and voltage barrier function
        env_config["mode"] = 'distributed'
        env_config["voltage_barrier_type"] = 'l1'
        env_config["data_path"] = os.path.join(global_data_source_path, "marllib/patch/dpn/var_voltage_control/data",
                                               net_topology)
        self.env = VoltageControl(env_config)

        self.action_space = Box(self.env.action_space.low, self.env.action_space.high, shape=(1,))
        self.observation_space = GymDict({
            "obs": Box(-100.0, 100.0, shape=(self.env.get_obs_size(),), ),
            "state": Box(-100.0, 100.0, shape=(self.env.get_state_size(),), ),
        })

        self.num_agents = self.env.get_num_of_agents()
        self.agents = ["agent_{}".format(i) for i in range(self.num_agents)]
        env_config["map_name"] = net_topology
        self.env_config = env_config

    def reset(self):
        o, s = self.env.reset()
        obs = {}
        for index, agent in enumerate(self.agents):
            obs[agent] = {
                "obs": np.float32(o[index]),
                "state": np.float32(s),
            }
        return obs

    def step(self, action_dict):
        action = [value[0] for value in action_dict.values()]
        r, d, info = self.env.step(action)
        o = self.env.get_obs()
        s = self.env.get_state()
        rewards = {}
        obs = {}
        for index, agent in enumerate(self.agents):
            obs[agent] = {
                "obs": np.float32(o[index]),
                "state": np.float32(s),
            }
            rewards[agent] = r
        dones = {"__all__": d}
        return obs, rewards, dones, {}

    def close(self):
        self.env.close()

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env_config["episode_limit"],
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
