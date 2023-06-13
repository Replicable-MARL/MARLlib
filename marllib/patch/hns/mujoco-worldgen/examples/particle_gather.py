import numpy as np
import gym
from gym.spaces import Box, Dict

from mujoco_worldgen import Floor, WorldBuilder, Geom, ObjFromXML, WorldParams, Env


def update_obs_space(env, delta):
    spaces = env.observation_space.spaces.copy()
    for key, shape in delta.items():
        spaces[key] = Box(-np.inf, np.inf, shape, np.float32)
    return Dict(spaces)


def rand_pos_on_floor(sim, n=1):
    world_size = sim.model.geom_size[sim.model.geom_name2id('floor0')] * 2
    new_pos = np.random.uniform(np.array([[0.2, 0.2] for _ in range(n)]),
                                np.array([world_size[:2] - 0.2 for _ in range(n)]))
    return new_pos


class GatherEnv(Env):
    def __init__(self, n_food=3, horizon=200, n_substeps=10,
                 floorsize=4., deterministic_mode=False):
        super().__init__(get_sim=self._get_sim,
                         get_obs=self._get_obs,
                         action_space=(-1.0, 1.0),
                         horizon=horizon,
                         deterministic_mode=deterministic_mode)
        self.n_food = n_food
        self.horizon = horizon
        self.n_substeps = n_substeps
        self.floorsize = floorsize

    def _get_obs(self, sim):
        qpos = sim.data.qpos.copy()
        qvel = sim.data.qvel.copy()
        qpos_qvel = np.concatenate([qpos, qvel], -1)
        return {'qpos': qpos, 'qvel': qvel, 'qpos_qvel': qpos_qvel}

    def _get_sim(self, seed):
        if self.sim is None:
            self.sim = self._get_new_sim(seed)

        self.sim.data.qpos[0:2] = rand_pos_on_floor(self.sim)
        return self.sim

    def _get_new_sim(self, seed):
        world_params = WorldParams(size=(self.floorsize, self.floorsize, 2.5),
                                   num_substeps=self.n_substeps)
        builder = WorldBuilder(world_params, seed)
        floor = Floor()
        builder.append(floor)
        # Walls
        wallsize = 0.1
        wall = Geom('box', (wallsize, self.floorsize, 0.5), name="wall1")
        wall.mark_static()
        floor.append(wall, placement_xy=(0, 0))
        wall = Geom('box', (wallsize, self.floorsize, 0.5), name="wall2")
        wall.mark_static()
        floor.append(wall, placement_xy=(1, 0))
        wall = Geom('box', (self.floorsize - wallsize*2, wallsize, 0.5), name="wall3")
        wall.mark_static()
        floor.append(wall, placement_xy=(1/2, 0))
        wall = Geom('box', (self.floorsize - wallsize*2, wallsize, 0.5), name="wall4")
        wall.mark_static()
        floor.append(wall, placement_xy=(1/2, 1))
        # Add agents
        obj = ObjFromXML("particle", name="agent0")
        floor.append(obj)
        obj.mark(f"object0")
        # Add food sites
        for i in range(self.n_food):
            floor.mark(f"food{i}", (.5, .5, 0.05), rgba=(0., 1., 0., 1.))
        sim = builder.get_sim()

        # Cache constants for quicker lookup later
        self.food_ids = np.array([sim.model.site_name2id(f'food{i}') for i in range(self.n_food)])
        return sim


class FoodHealthWrapper(gym.Wrapper):
    '''
        Adds food health to underlying env.
        Manages food levels.
        Randomizes food positions.
    '''
    def __init__(self, env, max_food_health=10):
        super().__init__(env)
        self.unwrapped.max_food_health = self.max_food_health = max_food_health
        self.unwrapped.max_food_size = self.max_food_size = 0.1
        self.observation_space = update_obs_space(env,
                                                  {'food_obs': (self.unwrapped.n_food, 4),
                                                   'food_pos': (self.unwrapped.n_food, 3),
                                                   'food_health': (self.unwrapped.n_food, 1)})

    def reset(self):
        obs = self.env.reset()

        # Reset food healths
        self.unwrapped.food_healths = np.array([self.max_food_health
                                                for _ in range(self.unwrapped.n_food)])
        # Randomize food positions
        new_pos = rand_pos_on_floor(self.unwrapped.sim, self.unwrapped.n_food)
        sites_offset = (self.unwrapped.sim.data.site_xpos -
                        self.unwrapped.sim.model.site_pos).copy()
        self.unwrapped.sim.model.site_pos[self.unwrapped.food_ids, :2] = \
            new_pos - sites_offset[self.unwrapped.food_ids, :2]

        # Reset food size
        self.unwrapped.sim.model.site_size[self.unwrapped.food_ids] = self.max_food_size

        return self.observation(obs)

    def observation(self, obs):
        # Add food position and healths to obersvations
        food_pos = self.unwrapped.sim.data.site_xpos[self.unwrapped.food_ids]
        food_health = self.unwrapped.food_healths
        obs['food_pos'] = food_pos
        obs['food_health'] = np.expand_dims(food_health, 1)
        obs['food_obs'] = np.concatenate([food_pos, np.expand_dims(food_health, 1)], 1)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        assert np.all(self.unwrapped.food_healths >= 0), \
            f"There is a food health below 0: {self.unwrapped.food_healths}"
        obs = self.observation(obs)
        return obs, rew, done, info


class ProcessEatFood(gym.Wrapper):
    """
        Manage food health. Resize food based on health.
        Expects a binary vector as input detailing which
    """
    def __init__(self, env, eat_thresh=0.7):
        super().__init__(env)
        self.n_food = self.unwrapped.n_food
        self.eat_thresh = eat_thresh

    def reset(self):
        return self.env.reset()

    def observation(self, obs):
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = self.observation(obs)

        # Eat food that is close enough
        agent_food_diff = obs['food_pos'] - np.expand_dims(obs['qpos'], axis=0)
        dist_to_food = np.linalg.norm(agent_food_diff, axis=-1)
        eat = np.logical_and(dist_to_food < self.eat_thresh, self.unwrapped.food_healths > 0)

        # Update food healths and sizes
        self.unwrapped.food_healths = self.unwrapped.food_healths - eat
        health_diff = np.expand_dims(eat, 1)
        size_diff = health_diff * (self.unwrapped.max_food_size / self.unwrapped.max_food_health)
        size = self.unwrapped.sim.model.site_size[self.unwrapped.food_ids] - size_diff
        size = np.maximum(0, size)
        self.unwrapped.sim.model.site_size[self.unwrapped.food_ids] = size

        rew += np.sum(eat)
        return obs, rew, done, info


def make_env(n_food=3, horizon=50, floorsize=4.):
    env = GatherEnv(horizon=horizon, floorsize=floorsize, n_food=n_food)
    env.reset()
    env = FoodHealthWrapper(env)
    env = ProcessEatFood(env)
    env.reset()
    return env
