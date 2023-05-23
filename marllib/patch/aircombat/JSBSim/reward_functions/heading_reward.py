import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class HeadingReward(BaseRewardFunction):
    """
    Measure the difference between the current heading and the target heading
    """
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_heading', '_alt', '_roll', '_speed']]

    def get_reward(self, task, env, agent_id):
        """
        Reward is built as a geometric mean of scaled gaussian rewards for each relevant variable

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """

        heading_error_scale = 5.0  # degrees
        heading_r = math.exp(-((env.agents[agent_id].get_property_value(c.delta_heading) / heading_error_scale) ** 2))

        alt_error_scale = 15.24  # m
        alt_r = math.exp(-((env.agents[agent_id].get_property_value(c.delta_altitude) / alt_error_scale) ** 2))

        roll_error_scale = 0.35  # radians ~= 20 degrees
        roll_r = math.exp(-((env.agents[agent_id].get_property_value(c.attitude_roll_rad) / roll_error_scale) ** 2))

        speed_error_scale = 24  # mps (~10%)
        speed_r = math.exp(-((env.agents[agent_id].get_property_value(c.delta_velocities_u) / speed_error_scale) ** 2))

        reward = (heading_r * alt_r * roll_r * speed_r) ** (1 / 4)
        return self._process(reward, agent_id, (heading_r, alt_r, roll_r, speed_r))
