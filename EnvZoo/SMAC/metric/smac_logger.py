import argparse
import os

from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import Logger


class SMACLogger(Logger):
    """
    Logs results by simply printing out what you need.
    Currently we use smac_reporter instead
    """

    def on_result(self, result: dict):
        # Define, what should happen on receiving a `result` (dict).
        reward_mean = result["episode_reward_mean"]
        reward_max = result["episode_reward_max"]
        reward_min = result["episode_reward_min"]

        custom_metrics_result = result["custom_metrics"]
        win_mean = custom_metrics_result["battle_win_rate_mean"]
        win_max = custom_metrics_result["battle_win_rate_max"]
        win_min = custom_metrics_result["battle_win_rate_min"]

        print("Battle Win Rate\t: max {win_max}, mean {win_mean}, min {win_min}".format(win_max=win_max,
                                                                                        win_mean=win_mean,
                                                                                        win_min=win_min))

        print("Reward\t: max {reward_max}, mean {reward_mean}, min {reward_min}".format(reward_max=reward_max,
                                                                                        reward_mean=reward_mean,
                                                                                        reward_min=reward_min))
