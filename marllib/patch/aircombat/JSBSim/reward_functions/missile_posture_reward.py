import numpy as np
from .reward_function_base import BaseRewardFunction


class MissilePostureReward(BaseRewardFunction):
    """
    MissilePostureReward
    Use the velocity attenuation
    """
    def __init__(self, config):
        super().__init__(config)
        self.previous_missile_v = None

    def reset(self, task, env):
        self.previous_missile_v = None
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        Reward is velocity attenuation of the missile

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward = 0
        missile_sim = env.agents[agent_id].check_missile_warning()
        if missile_sim is not None:
            missile_v = missile_sim.get_velocity()
            aircraft_v = env.agents[agent_id].get_velocity()
            if self.previous_missile_v is None:
                self.previous_missile_v = missile_v
            v_decrease = (np.linalg.norm(self.previous_missile_v) - np.linalg.norm(missile_v)) / 340 * self.reward_scale
            angle = np.dot(missile_v, aircraft_v) / (np.linalg.norm(missile_v) * np.linalg.norm(aircraft_v))
            if angle < 0:
                reward = angle / (max(v_decrease, 0) + 1)
            else:
                reward = angle * max(v_decrease, 0)
        else:
            self.previous_missile_v = None
            reward = 0
        self.reward_trajectory[agent_id].append([reward])
        return reward
