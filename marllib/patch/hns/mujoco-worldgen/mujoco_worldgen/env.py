import inspect
import logging
import hashlib

import gym
import numpy as np
from gym.spaces import Box, Tuple, Dict
from mujoco_py import MjSimState

from mujoco_worldgen.util.types import enforce_is_callable
from mujoco_worldgen.util.sim_funcs import (
    empty_get_info,
    flatten_get_obs,
    false_get_diverged,
    ctrl_set_action,
    zero_get_reward,
    )


logger = logging.getLogger(__name__)


class Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
    }

    def __init__(self,
                 get_sim,
                 get_obs=flatten_get_obs,
                 get_reward=zero_get_reward,
                 get_info=empty_get_info,
                 get_diverged=false_get_diverged,
                 set_action=ctrl_set_action,
                 action_space=None,
                 horizon=100,
                 start_seed=None,
                 deterministic_mode=False):
        """
        Env is a Gym environment subclass tuned for robotics learning
        research.

        Args:
        - get_sim (callable): a callable that returns an MjSim.
        - get_obs (callable): callable with an MjSim object as the sole
            argument and should return observations.
        - set_action (callable): callable which takes an MjSim object and
            updates its data and buffer directly.
        - get_reward (callable): callable which takes an MjSim object and
            returns a scalar reward.
        - get_info (callable): callable which takes an MjSim object and
            returns info (dictionary).
        - get_diverged (callable): callable which takes an MjSim object
            and returns a (bool, float) tuple. First value is True if
            simulator diverged and second value is the reward at divergence.
        - action_space: a space of allowed actions or a two-tuple of a ranges
            if number of actions is unknown until the simulation is instantiated
        - horizon (int): horizon of environment (i.e. max number of steps).
        - start_seed (int or string): seed for random state generator (None for random seed).
            Strings will be hashed.  A non-None value implies deterministic_mode=True.
            This argument allows us to run a deterministic series of goals/randomizations
            for a given policy.  Then applying the same seed to another policy will allow the
            comparison of results more accurately.  The reason a string is allowed is so
            that we can more easily find and share seeds that are farther from 0,
            which is the default starting point for deterministic_mode, and thus have
            more likelihood of getting a performant sequence of goals.
        """
        if (horizon is not None) and not isinstance(horizon, int):
            raise TypeError('horizon must be an int')

        self.get_sim = enforce_is_callable(get_sim, (
            'get_sim should be callable and should return an MjSim object'))
        self.get_obs = enforce_is_callable(get_obs, (
            'get_obs should be callable with an MjSim object as the sole '
            'argument and should return observations'))
        self.set_action = enforce_is_callable(set_action, (
            'set_action should be a callable which takes an MjSim object and '
            'updates its data and buffer directly'))
        self.get_reward = enforce_is_callable(get_reward, (
            'get_reward should be a callable which takes an MjSim object and '
            'returns a scalar reward'))
        self.get_info = enforce_is_callable(get_info, (
            'get_info should be a callable which takes an MjSim object and '
            'returns a dictionary'))
        self.get_diverged = enforce_is_callable(get_diverged, (
            'get_diverged should be a callable which takes an MjSim object '
            'and returns a (bool, float) tuple. First value is whether '
            'simulator is diverged (or done) and second value is the reward at '
            'that time.'))

        self.sim = None
        self.horizon = horizon
        self.t = None
        self.deterministic_mode = deterministic_mode

        # Numpy Random State
        if isinstance(start_seed, str):
            start_seed = int(hashlib.sha1(start_seed.encode()).hexdigest(), 16) % (2**32)
            self.deterministic_mode = True
        elif isinstance(start_seed, int):
            self.deterministic_mode = True
        else:
            start_seed = 0 if self.deterministic_mode else np.random.randint(2**32)
        self._random_state = np.random.RandomState(start_seed)
        # Seed that will be used on next _reset()
        self._next_seed = start_seed
        # Seed that was used in last _reset()
        self._current_seed = None

        # For rendering
        self.viewer = None

        # These are required by Gym
        self._action_space = action_space
        self._observation_space = None
        self._spec = Spec(max_episode_steps=horizon, timestep_limit=horizon)
        self._name = None

    # This is to mitigate issues with old/new envs
    @property
    def unwrapped(self):
        return self

    @property
    def name(self):
        if self._name is None:
            name = str(inspect.getfile(self.get_sim))
            if name.endswith(".py"):
                name = name[:-3]
            self._name = name
        return self._name

    def set_state(self, state, call_forward=True):
        """
        Sets the state of the enviroment to the given value. It does not
        set time.

        Warning: This only sets the MuJoCo state by setting qpos/qvel
            (and the user-defined state "udd_state"). It doesn't set
            the state of objects which don't have joints.

        Args:
        - state (MjSimState): desired state.
        - call_forward (bool): if True, forward simulation after setting
            state.
        """
        if not isinstance(state, MjSimState):
            raise TypeError("state must be an MjSimState")
        if self.sim is None:
            raise EmptyEnvException(
                "You must call reset() or reset_to_state() before setting the "
                "state the first time")

        # Call forward to write out values in the MuJoCo data.
        # Note: if udd_callback is set on the MjSim instance, then the
        # user will need to call forward() manually before calling step.
        self.sim.set_state(state)
        if call_forward:
            self.sim.forward()

    def get_state(self):
        """
        Returns a copy of the current environment state.

        Returns:
        - state (MjSimState): state of the environment's MjSim object.
        """
        if self.sim is None:
            raise EmptyEnvException(
                "You must call reset() or reset_to_state() before accessing "
                "the state the first time")
        return self.sim.get_state()

    def get_xml(self):
        '''
        :return: full state of the simulator serialized as XML (won't contain
                 meshes, textures, and data information).
        '''
        return self.sim.model.get_xml()

    def get_mjb(self):
        '''
        :return: full state of the simulator serialized as mjb.
        '''
        return self.sim.model.get_mjb()

    def reset_to_state(self, state, call_forward=True):
        """
        Reset to given state.

        Args:
        - state (MjSimState): desired state.
        """
        if not isinstance(state, MjSimState):
            raise TypeError(
                "You must reset to an explicit state (MjSimState).")

        if self.sim is None:
            if self._current_seed is None:
                self._update_seed()

            self.sim = self.get_sim(self._current_seed)
        else:
            # Ensure environment state not captured in MuJoCo's qpos/qvel
            # is reset to the state defined by the model.
            self.sim.reset()

        self.set_state(state, call_forward=call_forward)

        self.t = 0
        return self._reset_sim_and_spaces()

    def _update_seed(self, force_seed=None):
        if force_seed is not None:
            self._next_seed = force_seed
        self._current_seed = self._next_seed
        assert self._current_seed is not None
        # if in deterministic mode, then simply increment seed, otherwise randomize
        if self.deterministic_mode:
            self._next_seed = self._next_seed + 1
        else:
            self._next_seed = np.random.randint(2**32)
        # immediately update the seed in the random state object
        self._random_state.seed(self._current_seed)

    @property
    def current_seed(self):
        # Note: this is a property rather than just instance variable
        # for legacy and backwards compatibility reasons.
        return self._current_seed

    def _reset_sim_and_spaces(self):
        obs = self.get_obs(self.sim)

        # Mocaps are defined by 3-dim position and 4-dim quaternion
        if isinstance(self._action_space, tuple):
            assert len(self._action_space) == 2
            self._action_space = Box(
                self._action_space[0], self._action_space[1],
                (self.sim.model.nmocap * 7 + self.sim.model.nu, ), np.float32)
        elif self._action_space is None:
            self._action_space = Box(
                -np.inf, np.inf, (self.sim.model.nmocap * 7 + self.sim.model.nu, ), np.float32)
        self._action_space.flatten_dim = np.prod(self._action_space.shape)

        self._observation_space = gym_space_from_arrays(obs)
        if self.viewer is not None:
            self.viewer.update_sim(self.sim)

        return obs

    #
    # Custom pickling
    #

    def __getstate__(self):
        excluded_attrs = frozenset(
            ("sim", "viewer", "_monitor"))
        attr_values = {k: v for k, v in self.__dict__.items()
                       if k not in excluded_attrs}
        if self.sim is not None:
            attr_values['sim_state'] = self.get_state()
        return attr_values

    def __setstate__(self, attr_values):
        for k, v in attr_values.items():
            if k != 'sim_state':
                self.__dict__[k] = v

        self.sim = None
        self.viewer = None
        if 'sim_state' in attr_values:
            if self.sim is None:
                assert self._current_seed is not None
                self.sim = self.get_sim(self._current_seed)
            self.set_state(attr_values['sim_state'])
            self._reset_sim_and_spaces()

        return self

    def logs(self):
        logs = []
        if hasattr(self.env, 'logs'):
            logs += self.env.logs()
        return logs

    #
    # GYM REQUIREMENTS: these are methods required to be compatible with Gym
    #

    @property
    def action_space(self):
        if self._action_space is None:
            raise EmptyEnvException(
                "You have to reset environment before accessing action_space.")
        return self._action_space

    @property
    def observation_space(self):
        if self._observation_space is None:
            raise EmptyEnvException(
                "You have to reset environment before accessing "
                "observation_space.")
        return self._observation_space

    def reset(self, force_seed=None):
        self._update_seed(force_seed=force_seed)

        # get sim with current seed
        self.sim = self.get_sim(self._current_seed)

        # init sim
        self.sim.forward()
        self.t = 0
        self.sim.data.time = 0.0
        return self._reset_sim_and_spaces()

    def seed(self, seed=None):
        """
        Use `env.seed(some_seed)` to set the seed that'll be used in
        `env.reset()`. More specifically, this is the seed that will
        be passed into `env.get_sim` during `env.reset()`. The seed
        will then be incremented in consequent calls to `env.reset()`.
        For example:

            env.seed(0)
            env.reset() -> gives seed(0) world
            env.reset() -> gives seed(1) world
            ...
            env.seed(0)
            env.reset() -> gives seed(0) world
        """
        if isinstance(seed, list):
            # Support list of seeds as required by Gym.
            assert len(seed) == 1, "Only a single seed supported."
            self._next_seed = seed[0]
        elif isinstance(seed, int):
            self._next_seed = seed
        elif seed is not None:
            # If seed is None, we just return current seed.
            raise ValueError("Seed must be an integer.")

        # Return list of seeds to conform to Gym specs
        return [self._next_seed]

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.minimum(action, self.action_space.high)
        action = np.maximum(action, self.action_space.low)
        assert self.action_space.contains(action), (
            'Action should be in action_space:\nSPACE=%s\nACTION=%s' %
            (self.action_space, action))
        self.set_action(self.sim, action)
        self.sim.step()
        # Need to call forward() so that sites etc are updated,
        # since they're used in the reward computations.
        self.sim.forward()
        self.t += 1

        reward = self.get_reward(self.sim)
        if not isinstance(reward, float):
            raise TypeError("The return value of get_reward must be a float")

        obs = self.get_obs(self.sim)
        diverged, divergence_reward = self.get_diverged(self.sim)

        if not isinstance(diverged, bool):
            raise TypeError(
                "The first return value of get_diverged must be boolean")
        if not isinstance(divergence_reward, float):
            raise TypeError(
                "The second return value of get_diverged must be float")

        if diverged:
            done = True
            if divergence_reward is not None:
                reward = divergence_reward
        elif self.horizon is not None:
            done = (self.t >= self.horizon)
        else:
            done = False

        info = self.get_info(self.sim)
        info["diverged"] = divergence_reward
        # Return value as required by Gym
        return obs, reward, done, info

    def observe(self):
        """ Gets a new observation from the environment. """
        self.sim.forward()
        return self.get_obs(self.sim)

    def render(self, mode='human', close=False):
        if close:
            # TODO: actually close the inspection viewer
            return
        assert self.sim is not None, \
            "Please reset environment before render()."
        if mode == 'human':
            # Use a nicely-interactive version of the mujoco viewer
            if self.viewer is None:
                # Inline import since this is only relevant on platforms
                # which have GLFW support.
                from mujoco_py.mjviewer import MjViewer  # noqa
                self.viewer = MjViewer(self.sim)
            self.viewer.render()
        elif mode == 'rgb_array':
            return self.sim.render(500, 500)
        else:
            raise ValueError("Unsupported mode %s" % mode)


class EmptyEnvException(Exception):
    pass

# Helpers
###############################################################################


class Spec(object):
    # required by gym.wrappers.Monitor

    def __init__(self, max_episode_steps=np.inf, timestep_limit=np.inf):
        self.id = "worldgen.env"
        self.max_episode_steps = max_episode_steps
        self.timestep_limit = timestep_limit


def gym_space_from_arrays(arrays):
    if isinstance(arrays, np.ndarray):
        ret = Box(-np.inf, np.inf, arrays.shape, np.float32)
        ret.flatten_dim = np.prod(ret.shape)
    elif isinstance(arrays, (tuple, list)):
        ret = Tuple([gym_space_from_arrays(arr) for arr in arrays])
    elif isinstance(arrays, dict):
        ret = Dict(dict([(k, gym_space_from_arrays(v)) for k, v in arrays.items()]))
    else:
        raise TypeError("Array is of unsupported type.")
    return ret
