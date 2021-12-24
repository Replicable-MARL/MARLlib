from metadrive.envs.marl_envs import MultiAgentBottleneckEnv, MultiAgentParkingLotEnv, MultiAgentRoundaboutEnv, \
    MultiAgentIntersectionEnv, MultiAgentTollgateEnv

from ray.rllib.env import MultiAgentEnv
from collections import defaultdict
from metadrive.utils import norm
from MetaDrive.env.env_tools import *

NE_distance = 10

##################
### Bottleneck ###
##################
class Bottleneck_RLlib(MultiAgentBottleneckEnv, MultiAgentEnv):

    def __init__(self, config):
        super(Bottleneck_RLlib, self).__init__(config)
        self.__name__ = "Bottleneck"
        self.__qualname__ = "Bottleneck"


class Bottleneck_RLlib_Centralized_Critic(Bottleneck_RLlib):

    def __init__(self, config):
        super(Bottleneck_RLlib_Centralized_Critic, self).__init__(config)
        self.neighbours_distance = NE_distance
        self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))

    def step(self, actions):
        obs, reward, done, info = super(Bottleneck_RLlib_Centralized_Critic, self).step(actions)
        update_neighbours_map(self.distance_map, self.vehicles, reward, info, self.config)
        return obs, reward, done, info


##################
### ParkingLot ###
##################
class ParkingLot_RLlib(MultiAgentParkingLotEnv, MultiAgentEnv):

    def __init__(self, config):
        super(ParkingLot_RLlib, self).__init__(config)
        self.__name__ = "ParkingLot"
        self.__qualname__ = "ParkingLot"


class ParkingLot_RLlib_Centralized_Critic(ParkingLot_RLlib):

    def __init__(self, config):
        super(ParkingLot_RLlib_Centralized_Critic, self).__init__(config)
        self.neighbours_distance = NE_distance
        self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))

    def step(self, actions):
        obs, reward, done, info = super(ParkingLot_RLlib_Centralized_Critic, self).step(actions)
        update_neighbours_map(self.distance_map, self.vehicles, reward, info, self.config)
        return obs, reward, done, info


##################
### Intersection ###
##################
class Intersection_RLlib(MultiAgentIntersectionEnv, MultiAgentEnv):

    def __init__(self, config):
        super(Intersection_RLlib, self).__init__(config)
        self.__name__ = "Intersection"
        self.__qualname__ = "Intersection"


class Intersection_RLlib_Centralized_Critic(Intersection_RLlib):

    def __init__(self, config):
        super(Intersection_RLlib_Centralized_Critic, self).__init__(config)
        self.neighbours_distance = NE_distance
        self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))

    def step(self, actions):
        obs, reward, done, info = super(Intersection_RLlib_Centralized_Critic, self).step(actions)
        update_neighbours_map(self.distance_map, self.vehicles, reward, info, self.config)
        return obs, reward, done, info


##################
### Roundabout ###
##################
class Roundabout_RLlib(MultiAgentRoundaboutEnv, MultiAgentEnv):

    def __init__(self, config):
        super(Roundabout_RLlib, self).__init__(config)
        self.__name__ = "Roundabout"
        self.__qualname__ = "Roundabout"


class Roundabout_RLlib_Centralized_Critic(Roundabout_RLlib):

    def __init__(self, config):
        super(Roundabout_RLlib_Centralized_Critic, self).__init__(config)
        self.neighbours_distance = NE_distance
        self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))

    def step(self, actions):
        obs, reward, done, info = super(Roundabout_RLlib_Centralized_Critic, self).step(actions)
        update_neighbours_map(self.distance_map, self.vehicles, reward, info, self.config)
        return obs, reward, done, info


##################
### Tollgate ###
##################
class Tollgate_RLlib(MultiAgentTollgateEnv, MultiAgentEnv):

    def __init__(self, config):
        super(Tollgate_RLlib, self).__init__(config)
        self.__name__ = "Tollgate"
        self.__qualname__ = "Tollgate"


class Tollgate_RLlib_Centralized_Critic(Tollgate_RLlib):

    def __init__(self, config):
        super(Tollgate_RLlib_Centralized_Critic, self).__init__(config)
        self.neighbours_distance = NE_distance
        self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))

    def step(self, actions):
        obs, reward, done, info = super(Tollgate_RLlib_Centralized_Critic, self).step(actions)
        update_neighbours_map(self.distance_map, self.vehicles, reward, info, self.config)
        return obs, reward, done, info
