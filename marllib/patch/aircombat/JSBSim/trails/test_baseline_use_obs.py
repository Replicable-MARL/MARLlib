from abc import ABC
import sys
import os
# Deal with import error
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from envs.JSBSim.envs import SingleCombatEnv
from envs.JSBSim.utils.utils import get_root_dir
from envs.JSBSim.model.baseline_actor import BaselineActor

class BaselineAgent(ABC):
    def __init__(self, agent_id) -> None:
        self.model_path = get_root_dir() + '/model/baseline_model.pt'
        self.actor = BaselineActor()
        self.actor.load_state_dict(torch.load(self.model_path))
        self.actor.eval()
        self.agent_id = agent_id
        self.reset()

    def reset(self):
        self.rnn_states = np.zeros((1, 1, 128))

    @abstractmethod
    def set_delta_value(self, observation):
        raise NotImplementedError

    def get_observation(self, observation, delta_value):
        '''
        construct baseline observation from task observation 

        Baseline  observation:
        #  0. ego delta altitude      (unit: 1km)
        #  1. ego delta heading       (unit rad)
        #  2. ego delta velocities_u  (unit: mh)
        #  3. ego_altitude            (unit: 5km)
        #  4. ego_roll_sin
        #  5. ego_roll_cos
        #  6. ego_pitch_sin
        #  7. ego_pitch_cos
        #  8. ego_body_v_x            (unit: mh)
        #  9. ego_body_v_y            (unit: mh)
        #  10. ego_body_v_z           (unit: mh)
        #  11. ego_vc                 (unit: mh)
        '''
        norm_obs = np.zeros(12)
        norm_obs[:3] = delta_value
        norm_obs[3:12] = observation[:9]
        norm_obs = np.expand_dims(norm_obs, axis=0)  # dim: (1,12)
        return norm_obs

    def get_action(self, observation):
        delta_value = self.set_delta_value(observation[self.agent_id])
        obs = self.get_observation(observation[self.agent_id], delta_value)
        _action, self.rnn_states = self.actor(obs, self.rnn_states)
        action = _action.detach().cpu().numpy().squeeze()
        return action


class PursueAgent(BaselineAgent):
    def __init__(self, agent_id) -> None:
        super().__init__(agent_id)

    def set_delta_value(self, observation):
        delta_altitude = observation[10]
        delta_heading = observation[14]*observation[11]
        delta_velocity = observation[9]
        return np.array([delta_altitude, delta_heading, delta_velocity])


class ManeuverAgent(BaselineAgent):
    def __init__(self, agent_id, maneuver: Literal['l', 'r', 'n']) -> None:
        super().__init__(agent_id)
        self.turn_interval = 7         # unit: s
        self.env_time_interval = 0.2   # unit: 0.2s
        self.dodge_missile = True      # start turn when missile is detected, if set true
        if maneuver == 'l':
            self.delta_heading_list = [0, 0, 0, 0]
        elif maneuver == 'r':
            self.delta_heading_list = [np.pi/2, 0, 0, 0]
        elif maneuver == 'n':
            self.delta_heading_list = [np.pi/2, np.pi/2, 0, 0]

        self.target_altitude_list = [6096] * 4
        self.target_velocity_list = [243] * 4

    def reset(self):
        self.step = 0
        self.rnn_states = np.zeros((1, 1, 128))

    def set_delta_value(self, observation):
        step_list = np.arange(1, len(self.delta_heading_list)+1) * self.turn_interval / self.env_time_interval
        if not self.dodge_missile or (len(observation) > 15 and observation[15] != 0):
            for i, interval in enumerate(step_list):
                if self.step <= interval:
                    break
            delta_heading = self.delta_heading_list[i]
            delta_altitude = (self.target_altitude_list[i] - observation[0]*5000) / 1000
            delta_velocity = (self.target_velocity_list[i] - observation[5]*340) / 340
            self.step += 1
        else:
            delta_heading = 0
            delta_altitude = 0
            delta_velocity = 0

        return np.array([delta_altitude, delta_heading, delta_velocity])


def test_maneuver():
    env = SingleCombatEnv(config_name='1v1/DodgeMissile/Selfplay')
    env.seed(0)
    obs = env.reset()
    env.render()
    agent0 = ManeuverAgent(agent_id=0, maneuver='n')
    agent1 = PursueAgent(agent_id=1)
    reward_list = []
    step = 0
    bloods_list = []
    while True:
        actions = [agent0.get_action(obs), agent1.get_action(obs)]
        obs, reward, done, info = env.step(actions)
        env.render()
        bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
        print(f"step:{step}, bloods:{bloods}")
        reward_list.append(reward[0])
        if np.array(done).all():
            print(info)
            break
        step += 1
    
    env.seed(0)
    obs = env.reset()
    env.render()
    agent0 = ManeuverAgent(agent_id=0, maneuver='n')
    agent1 = PursueAgent(agent_id=1)
    reward_list = []
    step = 0
    bloods_list = []
    while True:
        actions = [agent0.get_action(obs), agent1.get_action(obs)]
        obs, reward, done, info = env.step(actions)
        env.render()
        bloods = [env.agents[agent_id].bloods for agent_id in env.agents.keys()]
        print(f"step:{step}, bloods:{bloods}")
        reward_list.append(reward[0])
        if np.array(done).all():
            print(info)
            break
        step += 1
    # plt.plot(reward_list)
    # plt.savefig('rewards.png')


if __name__ == '__main__':
    test_maneuver()