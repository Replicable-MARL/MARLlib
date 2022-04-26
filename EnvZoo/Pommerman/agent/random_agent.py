from pommerman.agents import BaseAgent
from pommerman import characters
from pommerman.characters import Bomber
from pommerman import constants
from pommerman import utility


class RandomAgent(Bomber):
    """Code exactly same as The Random Agent"""

    def __init__(self, *args, **kwargs):
        super(RandomAgent, self).__init__(*args, **kwargs)

        # Keep track of recently visited uninteresting positions so that we
        # don't keep visiting the same places.
        self._recently_visited_positions = []
        self._recently_visited_length = 6
        # Keep track of the previous direction to help with the enemy standoffs.
        self._prev_direction = None

    def episode_end(self, reward):
        """This is called at the end of the episode to let the agent know that
        the episode has ended and what is the reward.

        Args:
          reward: The single reward scalar to this agent.
        """
        pass

    def init_agent(self, id_, game_type):
        super(RandomAgent, self).__init__(id_, game_type)

    @staticmethod
    def has_user_input():
        return False

    def shutdown(self):
        pass

    def act(self, obs, action_space):
        return action_space.sample()
