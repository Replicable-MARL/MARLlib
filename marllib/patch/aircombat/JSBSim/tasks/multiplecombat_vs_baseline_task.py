import numpy as np
from gym import spaces
from typing import Tuple
import torch

from .multiplecombat_task import MultipleCombatTask, HierarchicalMultipleCombatTask, HierarchicalMultipleCombatShootTask



class MultipleCombatVsBaselineTask(MultipleCombatTask):

    @property
    def num_agents(self) -> int:  # ally number
        agent_num = 0
        for key in self.config.aircraft_configs.keys():
            if "A" in key:
                agent_num += 1
        return agent_num

    def load_observation_space(self):
        aircraft_num = len(self.config.aircraft_configs)
        self.obs_length = 9 + (aircraft_num - 1) * 6
        self.observation_space = spaces.Box(low=-10, high=10., shape=(self.obs_length,))
        self.share_observation_space = spaces.Box(low=-10, high=10., shape=(aircraft_num * self.obs_length,))

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        if self.use_baseline and agent_id in env.enm_ids:
            action = self.baseline_agent.get_action(env.agents[agent_id])
            return action
        else:
            norm_act = np.zeros(4)
            norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
            norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
            norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
            norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
            return norm_act


class HierarchicalMultipleCombatVsBaselineTask(HierarchicalMultipleCombatTask):

    def normalize_action(self, env, agent_id, action):
        """Convert high-level action into low-level action.
        """
        if self.use_baseline and agent_id in env.enm_ids:
            action = self.baseline_agent.get_action(env.agents[agent_id])
            return action
        else:
            # generate low-level input_obs
            raw_obs = self.get_obs(env, agent_id)
            input_obs = np.zeros(12)
            # (1) delta altitude/heading/velocity
            input_obs[0] = self.norm_delta_altitude[action[0]]
            input_obs[1] = self.norm_delta_heading[action[1]]
            input_obs[2] = self.norm_delta_velocity[action[2]]
            # (2) ego info
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.expand_dims(input_obs, axis=0)
            # output low-level action
            _action, _rnn_states = self.lowlevel_policy(input_obs, self._inner_rnn_states[agent_id])
            action = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
            # normalize low-level action
            norm_act = np.zeros(4)
            norm_act[0] = action[0] / 20 - 1.
            norm_act[1] = action[1] / 20 - 1.
            norm_act[2] = action[2] / 20 - 1.
            norm_act[3] = action[3] / 58 + 0.4
            return norm_act

    def reset(self, env):
        """Task-specific reset, include reward function reset.
        """
        self._inner_rnn_states = {agent_id: np.zeros((1, 1, 128)) for agent_id in env.agents.keys()}
        return super().reset(env)
