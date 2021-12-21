from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.env.multi_agent_env import MultiAgentEnv as Rllib_MultiAgentEnv
from smac.env.starcraft2.starcraft2 import StarCraft2Env
import numpy as np
from absl import logging
from pysc2.lib import protocol
from s2clientprotocol import sc2api_pb2 as sc_pb
from gym.spaces import Dict, Discrete, Box


class StarCraft2Env_Rllib(StarCraft2Env, Rllib_MultiAgentEnv):
    """The StarCraft II environment for decentralised multi-agent
    micromanagement scenarios.
    """

    def __init__(self, map_name):

        map_name = map_name if isinstance(map_name, str) else map_name["map_name"]

        StarCraft2Env.__init__(self, map_name)
        Rllib_MultiAgentEnv.__init__(self)

        env_info = self.get_env_info()
        obs_shape = env_info['obs_shape']
        n_actions = env_info['n_actions']
        state_shape = env_info['state_shape']
        self.observation_space = Dict({
                                  "obs": Box(-2.0, 2.0, shape=(obs_shape,)),
                                  "state": Box(-2.0, 2.0, shape=(state_shape,)),
                                  "action_mask": Box(0.0, 1.0, shape=(n_actions,))
                              })
        self.action_space = Discrete(n_actions)

    # override
    def reset(self):
        """Reset the environment. Required after each full episode.
        Returns initial observations and states.
        """
        self._episode_steps = 0
        if self._episode_count == 0:
            # Launch StarCraft II
            self._launch()
        else:
            self._restart()

        # Information kept for counting the reward
        self.death_tracker_ally = np.zeros(self.n_agents)
        self.death_tracker_enemy = np.zeros(self.n_enemies)
        self.previous_ally_units = None
        self.previous_enemy_units = None
        self.win_counted = False
        self.defeat_counted = False

        self.last_action = np.zeros((self.n_agents, self.n_actions))

        if self.heuristic_ai:
            self.heuristic_targets = [None] * self.n_agents

        try:
            self._obs = self._controller.observe()
            self.init_units()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()

        if self.debug:
            logging.debug(
                "Started Episode {}".format(self._episode_count).center(
                    60, "*"
                )
            )

        obs_smac_api = self.get_obs()
        state_smac_api = self.get_state()
        obs_rllib = {}
        for agent_index in range(self.n_agents):
            obs_one_agent = obs_smac_api[agent_index]
            state_one_agent = state_smac_api
            action_mask_one_agent = np.array(self.get_avail_agent_actions(agent_index))
            obs_rllib[agent_index] = {
                "obs": obs_one_agent,
                "state": state_one_agent,
                "action_mask": action_mask_one_agent
            }

        return obs_rllib

    # override
    def step(self, actions):

        actions_int = [int(actions[agent_id]) for agent_id in actions.keys()]

        self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        # Collect individual actions
        sc_actions = []
        if self.debug:
            logging.debug("Actions".center(60, "-"))

        for a_id, action in enumerate(actions_int):
            if not self.heuristic_ai:
                sc_action = self.get_agent_action(a_id, action)
            else:
                sc_action, action_num = self.get_agent_action_heuristic(
                    a_id, action
                )
                actions[a_id] = action_num
            if sc_action:
                sc_actions.append(sc_action)

        # Send action request
        req_actions = sc_pb.RequestAction(actions=sc_actions)
        try:
            self._controller.actions(req_actions)
            # Make step in SC2, i.e. apply actions
            self._controller.step(self._step_mul)
            # Observe here so that we know if the episode is over.
            self._obs = self._controller.observe()
        except (protocol.ProtocolError, protocol.ConnectionError):
            self.full_restart()
            reward_rllib = {}
            for agent_index in range(self.n_agents):
                reward_rllib[agent_index] = 0
                dones = {"__all__": True}
            return self.obs_rllib, reward_rllib, dones, {}

        self._total_steps += 1
        self._episode_steps += 1

        # Update units
        game_end_code = self.update_units()

        terminated = False
        reward = self.reward_battle()
        info = {"battle_won": False}

        # count units that are still alive
        dead_allies, dead_enemies = 0, 0
        for _al_id, al_unit in self.agents.items():
            if al_unit.health == 0:
                dead_allies += 1
        for _e_id, e_unit in self.enemies.items():
            if e_unit.health == 0:
                dead_enemies += 1

        info["dead_allies"] = dead_allies
        info["dead_enemies"] = dead_enemies

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                self.battles_won += 1
                self.win_counted = True
                info["battle_won"] = True
                if not self.reward_sparse:
                    reward += self.reward_win
                else:
                    reward = 1
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                if not self.reward_sparse:
                    reward += self.reward_defeat
                else:
                    reward = -1

        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1

        if self.debug:
            logging.debug("Reward = {}".format(reward).center(60, "-"))

        if terminated:
            self._episode_count += 1

        if self.reward_scale:
            reward /= self.max_reward / self.reward_scale_rate

        self.reward = reward

        obs_smac_api = self.get_obs()
        state_smac_api = self.get_state()
        obs_rllib = {}
        reward_rllib = {}
        for agent_index in range(self.n_agents):
            obs_one_agent = obs_smac_api[agent_index]
            state_one_agent = state_smac_api
            action_mask_one_agent = np.array(self.get_avail_agent_actions(agent_index))
            obs_rllib[agent_index] = {
                "obs": obs_one_agent,
                "state": state_one_agent,
                "action_mask": action_mask_one_agent
            }
            reward_rllib[agent_index] = reward

        dones = {"__all__": terminated}

        self.obs_rllib = obs_rllib

        return obs_rllib, reward_rllib, dones, {}

    # override
    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        env_info["agent_features"] = self.ally_state_attr_names
        env_info["enemy_features"] = self.enemy_state_attr_names
        return env_info
