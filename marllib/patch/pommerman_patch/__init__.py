# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
