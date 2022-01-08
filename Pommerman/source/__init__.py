'''Entry point into the pommerman module'''
import gym
import inspect
from . import agents
from . import configs
from . import constants
from . import forward_model
from . import helpers
from . import utility
from . import network

gym.logger.set_level(40)
REGISTRY = None


def _register():
    global REGISTRY
    REGISTRY = []
    for name, f in inspect.getmembers(configs, inspect.isfunction):
        if not name.endswith('_env'):
            continue

        config = f()
        gym.envs.registration.register(
            id=config['env_id'],
            entry_point=config['env_entry_point'],
            kwargs=config['env_kwargs']
        )
        REGISTRY.append(config['env_id'])


# Register environments with gym
_register()

def make(config_id, agent_list, game_state_file=None, render_mode='human'):
    '''Makes the pommerman env and registers it with gym'''
    assert config_id in REGISTRY, "Unknown configuration '{}'. " \
        "Possible values: {}".format(config_id, REGISTRY)
    env = gym.make(config_id)

    for id_, agent in enumerate(agent_list):
        # assert isinstance(agent, agents.BaseAgent)
        # NOTE: This is IMPORTANT so that the agent character is initialized
        agent.init_agent(id_, env.spec._kwargs['game_type'])

    env.set_agents(agent_list)
    env.set_init_game_state(game_state_file)
    env.set_render_mode(render_mode)
    return env


from . import cli
