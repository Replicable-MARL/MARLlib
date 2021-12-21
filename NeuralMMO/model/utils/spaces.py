''' copied and modified from Neural-MMO RLlib_Wrapper '''

import gym
from ray.rllib.utils.spaces.flexdict import FlexDict
from neural_mmo.forge.trinity.dataframe import DataType
from NeuralMMO.model.utils.output import *


def observationSpace(config):
    obs = FlexDict(defaultdict(FlexDict))
    for entity in sorted(Stimulus.values()):
        nRows = entity.N(config)
        nContinuous = 0
        nDiscrete = 0

        for _, attr in entity:
            if attr.DISCRETE:
                nDiscrete += 1
            if attr.CONTINUOUS:
                nContinuous += 1

        obs[entity.__name__]['Continuous'] = gym.spaces.Box(
            low=-2 ** 20, high=2 ** 20, shape=(nRows, nContinuous),
            dtype=DataType.CONTINUOUS)

        obs[entity.__name__]['Discrete'] = gym.spaces.Box(
            low=0, high=4096, shape=(nRows, nDiscrete),
            dtype=DataType.DISCRETE)

    obs['Entity']['N'] = gym.spaces.Box(
        low=0, high=config.N_AGENT_OBS, shape=(1,),
        dtype=DataType.DISCRETE)
    return obs


def actionSpace(config):
    atns = FlexDict(defaultdict(FlexDict))
    for atn in sorted(Action.edges):
        for arg in sorted(atn.edges):
            n = arg.N(config)
            atns[atn][arg] = gym.spaces.Discrete(n)
    return atns
