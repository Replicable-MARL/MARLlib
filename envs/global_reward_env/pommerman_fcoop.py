import numpy as np
from envs.base_env.pommerman import RllibPommerman, get_obs_dict


"""
"OneVsOne-v0",
"PommeFFACompetition-v0",
"PommeTeamCompetition-v0",
"""


class RllibPommerman_FCOOP(RllibPommerman):

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

        r = 0
        for x in range(self.num_agents):
            if self.num_agents > 1:
                r += all_reward[self.neural_agent[x]]

        for x in range(self.num_agents):
            if self.num_agents > 1:
                # state_current_agent
                s_c_a = all_state[self.neural_agent[x]]
                obs_status = get_obs_dict(s_c_a)
                states["agent_%d" % x] = obs_status
                rewards["agent_%d" % x] = r
                infos["agent_%d" % x] = {}

            else:
                print("agent number must > 1")
                raise ValueError()

        dones = {"__all__": done}
        return states, rewards, dones, infos

