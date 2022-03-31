import os
import tempfile
import numpy as np
import argparse
import gym
import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from gym.spaces import Dict, Discrete, Box
import pommerman
from Pommerman.agent.trainable_place_holder_agent import PlaceHolderAgent
import sys


class RllibPommerman(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def __init__(self, env_config, agent_list):
        self.map = env_config["map"]
        self.env = pommerman.make(env_config["map"], agent_list)
        self.neural_agent = env_config["neural_agent_pos"]
        self.rule_agent = env_config["rule_agent_pos"]

        agent_num = 0
        for agent in agent_list:
            if type(agent) == PlaceHolderAgent:
                self.env.set_training_agent(agent.agent_id)
                agent_num += 1

        if "One" in self.map:  # for Map OneVsOne-v0
            map_size = 8
        else:
            map_size = 11

        self.action_space = self.env.action_space
        self.observation_space = Dict({
            "obs": Box(-100.0, 100.0, shape=(map_size, map_size, 5)),
            "status": Box(-100.0, 100.0, shape=(4,)),  # position + blast strength + can kick
        })

        self.num_agents = agent_num

    def reset(self):
        original_all_state = self.env.reset()
        self.state_store = original_all_state
        state = {}
        for x in range(self.num_agents):
            if self.num_agents > 1:
                # state_current_agent
                s_c_a = original_all_state[self.neural_agent[x]]
                obs_status = get_obs_dict(s_c_a)
                state["agent_%d" % x] = obs_status
            else:
                print("agent number must > 1")
                sys.exit()
        return state

    def step(self, action_dict):
        # fake action
        if self.map == "OneVsOne-v0":  # 2 agents map
            actions = [-1, -1, ]
        else:
            actions = [-1, -1, -1, -1]

        # actions for SimpleAgent (non-trainable):
        non_trainable_actions = self.env.act(self.state_store)
        if self.rule_agent == []:
            pass
        else:
            for index, rule_based_agent_number in enumerate(self.rule_agent):
                actions[rule_based_agent_number] = non_trainable_actions[index]

        for index, key in enumerate(action_dict.keys()):
            value = action_dict[key]
            trainable_agent_number = self.neural_agent[index]
            actions[trainable_agent_number] = value

        if -1 in actions:
            raise ValueError()

        all_state, all_reward, done, all_info = self.env.step(actions)
        self.state_store = all_state
        rewards = {}
        states = {}
        infos = {}

        for x in range(self.num_agents):
            if self.num_agents > 1:
                # state_current_agent
                s_c_a = all_state[self.neural_agent[x]]
                obs_status = get_obs_dict(s_c_a)
                states["agent_%d" % x] = obs_status
                rewards["agent_%d" % x] = all_reward[self.neural_agent[x]]
                infos["agent_%d" % x] = {}

            else:
                print("agent number must > 1")
                raise ValueError()

        dones = {"__all__": done}
        return states, rewards, dones, infos


def get_obs_dict(state_current_agent):
    obs = np.stack((state_current_agent["board"],
                    state_current_agent["bomb_blast_strength"],
                    state_current_agent["bomb_life"],
                    state_current_agent["bomb_moving_direction"],
                    state_current_agent["flame_life"]),
                   axis=2)
    position = np.array(state_current_agent["position"])
    blast_strength = np.array([state_current_agent["blast_strength"]])
    can_kick = np.array([1]) if state_current_agent["can_kick"] else np.array([0])
    status = np.concatenate([position, blast_strength, can_kick])

    return {"obs": obs.astype(np.float32), "status": status.astype(np.float32)}
