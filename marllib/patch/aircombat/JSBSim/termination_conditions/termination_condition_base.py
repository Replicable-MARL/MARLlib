import logging
from abc import ABC, abstractmethod


class BaseTerminationCondition(ABC):
    """
    Base TerminationCondition class
    Condition-specific get_termination method is implemented in subclasses
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_termination(self, task, env, agent_id, info={}):
        """
        Return whether the episode should terminate.
        Overwritten by subclasses.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (tuple): (done, success, info)
        """
        raise NotImplementedError

    def log(self, msg):
        logging.debug(msg)
