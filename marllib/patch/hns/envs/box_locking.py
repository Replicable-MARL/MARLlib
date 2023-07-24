import gym
import numpy as np
from marllib.patch.hns.wrappers.multi_agent import (SplitMultiAgentActions, SplitObservations,
                                           SelectKeysWrapper
                                        #    InitSetWrapper
                                           )
from marllib.patch.hns.wrappers.util import (DiscretizeActionWrapper, MaskActionWrapper,
                                    DiscardMujocoExceptionEpisodes,
                                    AddConstantObservationsWrapper,
                                    SpoofEntityWrapper, ConcatenateObsWrapper)
from marllib.patch.hns.wrappers.manipulation import (GrabObjWrapper, GrabClosestWrapper, TimeWrapper,
                                            LockObjWrapper, LockAllWrapper)
from marllib.patch.hns.wrappers.lidar import Lidar
from marllib.patch.hns.wrappers.line_of_sight import AgentAgentObsMask2D, AgentGeomObsMask2D
from marllib.patch.hns.wrappers.team import TeamMembership
from marllib.patch.hns.wrappers.util import NumpyArrayRewardWrapper
from marllib.patch.hns.modules.agents import Agents, AgentManipulation
from marllib.patch.hns.modules.walls import RandomWalls, WallScenarios
from marllib.patch.hns.modules.objects import Boxes, Ramps, LidarSites
from marllib.patch.hns.modules.world import FloorAttributes, WorldConstants
from marllib.patch.hns.modules.util import uniform_placement, center_placement, centerOrset_placement
from marllib.patch.hns.envs.base import Base
from marllib.patch.hns.envs.hide_and_seek import uniformOrset_placement,quadrant_placement,outside_quadrant_placement,quadrantOrset_placement
from marllib.patch.hns.wrappers.util import update_obs_space


class LockObjectsTask(gym.Wrapper):
    """
        Reward wrapper for the lock object family of tasks. The reward consists of four components:
        (1) A fixed reward for locking a new box;
        (2) A fixed penalty for unlocking a previously locked box;
        (3) A shaped reward proportional to the reduction in distance between the agent and its
            next target (which is either the next box that needs to be locked or the agent's
            spawning point)
        (4) A success reward that is received in every timestep during which the task is completed.
            E.g. in the 'all' task, the success reward is received in every timestep where all
            boxes are locked; but if the agent locks all boxes and later accidentally unlocks a box,
            it will stop receiving the reward until all boxes are locked again.
        Args:
            n_objs (int): number of objects
            task (str): {'all', 'order'} + ['-return']
                'all': success when all boxes are locked
                'order': success when boxes are locked in a specific order
                'xxx-return': after finishing the task of 'xxx', the agent needs to
                    return to the location it was spawned at the beginning of the episode.
            fixed_order (bool): if True, the order/selection of boxes to lock for a
                success will always be fixed
            obj_lock_obs_key (str): Observation key for which objects are currently locked.
            obj_pos_obs_key (str): Observation key for object positions
            act_lock_key (str): Action key for lock action
            agent_pos_key (str): Observation key for agent positions
            lock_reward (float): Reward for locking a box
            unlock_penalty (float): Penalty for unlocking a box
            shaped_reward_scale (float): Scales the shaped reward by this factor
            success_reward (float): This reward is received in every timestep during which
                the task is completed.
            return_threshold (float): In 'xxx-return' tasks, after finishing the base task
                the agent needs to return within this distance of its original spawning
                position in order for the task to be considered completed
    """
    def __init__(self, env, n_objs, task='all', fixed_order=False,
                 obj_lock_obs_key='obj_lock', obj_pos_obs_key='box_pos',
                 act_lock_key='action_glue', agent_pos_key='agent_pos',
                 lock_reward=5.0, unlock_penalty=10.0, shaped_reward_scale=1.0,
                 success_reward=1, return_threshold=0.1):
        super().__init__(env)
        self.n_objs = n_objs
        self.task = task or 'all'
        assert task in ['all', 'order', 'all-return', 'order-return'], (
            f'task {task} is currently not supported')
        self.need_return = 'return' in self.task
        self.return_threshold = return_threshold
        if self.need_return:
            self.task = self.task.replace('-return', '')
        self.n_agents = self.unwrapped.n_agents
        #assert self.n_agents == 1, 'The locking tasks only support 1 agent'
        self.agent_key = agent_pos_key
        self.obj_order = list(range(self.n_objs))
        self.fixed_order = fixed_order
        self.lock_key = obj_lock_obs_key
        self.pos_key = obj_pos_obs_key
        self.act_key = act_lock_key
        self.lock_reward = lock_reward
        self.unlock_penalty = unlock_penalty
        self.shaped_reward_scale = shaped_reward_scale
        self.success_reward = success_reward
        self.success = False
        self.objs_locked = np.zeros((n_objs, ), dtype=np.int8)
        self.spawn_pos = None
        self.spawn_pos_dist = None
        self.next_obj = None
        self.next_obj_dist = 0
        self.unlocked_objs = []
        self.observation_space = update_obs_space(self.env, {'agent_spawn_obs': (self.n_agents, 2)}) #

    def reset(self):
        if not self.fixed_order:
            np.random.shuffle(self.obj_order)
        self.objs_locked[:] = 0
        self.unlocked_objs = self.obj_order
        obs = self.env.reset()
        #self.spawn_pos = obs[self.agent_key][0, :2]
        # for multi agent
        self.spawn_pos = obs[self.agent_key][:, :2]
        self.spawn_pos_dist = 0
        self.next_obj, self.next_obj_dist = self._get_next_obj(obs)
        self.success = False
        obs['agent_spawn_obs'] = self.spawn_pos  #add spawn_pos to obs dict
        return obs

    def _get_next_obj(self, obs):
        '''
            Return the next object that needs to be locked & the distance to it.
        '''
        agent_pos = obs[self.agent_key][:, :2]
        if len(self.unlocked_objs) == 0:
            next_obj = None
            next_obj_dist = 0
        elif self.task == 'order':
            next_obj = self.unlocked_objs[0]
            next_obj_pos = obs[self.pos_key][next_obj, :2]
            next_obj_dist = np.linalg.norm(agent_pos - next_obj_pos)
        elif self.task == 'all':
            obj_dist = [(np.linalg.norm(obs[self.pos_key][i, :2] - agent_pos), i)
                        for i in self.unlocked_objs]
            next_obj_dist, next_obj = min(obj_dist)

        return next_obj, next_obj_dist

    def _get_lock_reward(self, curr_objs_locked, old_objs_locked):
        '''
            Calculates the locking reward / unlocking penalty
        '''
        n_new_lock = np.sum(np.logical_and(curr_objs_locked == 1, old_objs_locked == 0))
        n_new_unlock = np.sum(np.logical_and(curr_objs_locked == 0, old_objs_locked == 1))
        lock_reward = n_new_lock * self.lock_reward - n_new_unlock * self.unlock_penalty
        return lock_reward

    def _get_shaped_reward(self, new_next_obj, new_next_obj_dist, new_spawn_pos_dist):
        '''
            Calculates the shaped reward based on the change in distance from the target
        '''
        rew = 0
        if (self.next_obj is not None) and (new_next_obj == self.next_obj):
            rew += (self.next_obj_dist - new_next_obj_dist) * self.shaped_reward_scale
        elif ((self.next_obj is not None) and (new_next_obj != self.next_obj)):
            if self.objs_locked[self.next_obj] == 1:
                # previous target object locked
                rew += self.next_obj_dist * self.shaped_reward_scale
            else:
                # previously locked object unlocked
                rew -= new_next_obj_dist * self.shaped_reward_scale
        elif (self.next_obj is None) and (new_next_obj is not None):
            # previously locked object unlocked
            rew -= new_next_obj_dist * self.shaped_reward_scale
        elif (self.next_obj is None) and (new_next_obj is None):
            if self.need_return:
                # all objects locked; agent is rewarded for returning to its spawning point
                rew += (self.spawn_pos_dist - new_spawn_pos_dist) * self.shaped_reward_scal
        return rew

    def step(self, action):
        if self.task == 'order':
            """
                you can unlock any locked objs but only lock objs when all previous ones are locked
            """
            if len(self.unlocked_objs) > 1:
                action[self.act_key][:, self.unlocked_objs[1:]] = 0

        obs, rew, done, info = self.env.step(action)
        obs['agent_spawn_obs'] = self.spawn_pos #add spawn pos to obs dict
        curr_objs_locked = obs[self.lock_key].flatten().astype(np.int8)

        # rew += self._get_lock_reward(curr_objs_locked, old_objs_locked=self.objs_locked)

        self.objs_locked = curr_objs_locked
        self.unlocked_objs = [i for i in self.obj_order if self.objs_locked[i] == 0]

        # use sparse reward
        '''
        new_next_obj, new_next_obj_dist = self._get_next_obj(obs)
        agent_pos = obs[self.agent_key][:, :2]
        new_spawn_pos_dist = np.linalg.norm(agent_pos - self.spawn_pos)
        rew += self._get_shaped_reward(new_next_obj, new_next_obj_dist, new_spawn_pos_dist)

        self.spawn_pos_dist = new_spawn_pos_dist
        self.next_obj_dist = new_next_obj_dist
        self.next_obj = new_next_obj
        '''
        agent_pos = obs[self.agent_key][:, :2]
        spawn_pos_dist_byone = np.linalg.norm(agent_pos - self.spawn_pos, axis=1)
        new_spawn_pos_dist = np.linalg.norm(agent_pos - self.spawn_pos)
        agent_return = (spawn_pos_dist_byone <= self.return_threshold)
        
        n_unlocked = len(self.unlocked_objs)
        n_return = 0

        success_lock = False
        success_return = False
        if n_unlocked == 0:
            # reward for successfully locking all boxes
            rew += self.success_reward * 0.2
            success_lock = True
            if self.need_return:
                n_return = np.sum(agent_return)
                if agent_return.all():
                    success_return = True
                    rew += 1
        self.success = success_lock and ((not self.need_return) or success_return)
        
        if self.need_return:
            lock_rate = 1.0 - n_unlocked / self.n_objs
            return_rate = n_return / self.n_agents
            info['lock_rate'] = lock_rate 
            info['return_rate'] = return_rate 
        else:
            lock_rate = 1.0 - n_unlocked / self.n_objs
            info['lock_rate'] = lock_rate 
            info['return_rate'] = 0 
            
        info['success'] = self.success

        return obs, rew, done, info

def tri_placement(tri_room_idx):
    '''
        This function expects the wall scenario to be 'var_tri'
        Returns a placement function that randomly places objects in the room
        with index tri_room_idx
    '''
    def placement(grid, obj_size, metadata, random_state):
        assert 'tri_room_grid_cell_range' in metadata
        x_rag, y_rag = metadata['tri_room_grid_cell_range'][tri_room_idx]
        pos = np.array([random_state.randint(x_rag[0], x_rag[1] - obj_size[0]),
                        random_state.randint(y_rag[0], y_rag[1] - obj_size[1])])
        return pos

    return placement

def rotate_tri_placement(grid, obj_size, metadata, random_state):
    '''
        This function expects the wall scenario to be 'var_tri'.
        It places objects equally among the three rooms, so that any room has
        contains at most 1 more object than any other room.
    '''
    if 'tri_placement_rotation' not in metadata:
        metadata['tri_placement_rotation'] = []
    filled_rooms = metadata['tri_placement_rotation']
    if len(filled_rooms) == 3:
        filled_rooms = []
    available_rooms = [i for i in range(3) if i not in filled_rooms]
    n_available_rooms = len(available_rooms)
    next_room = available_rooms[random_state.randint(0, 10000) % n_available_rooms]
    filled_rooms.append(next_room)
    metadata['tri_placement_rotation'] = filled_rooms
    return tri_placement(next_room)(grid, obj_size, metadata, random_state)

def make_env(args):
    return BoxLockingEnv(args)

def BoxLockingEnv(args, n_substeps=15, horizon=60, deterministic_mode=True,
             floor_size=6.0, grid_size=30, door_size=2,
             n_agents=2, fixed_agent_spawn=False,
             lock_box=True, grab_box=True, grab_selective=False,
             lock_type='all_lock_team_specific',
             lock_grab_radius=0.25, grab_exclusive=False, grab_out_of_vision=False,
             lock_out_of_vision=False,
             box_floor_friction=0.2, other_friction=0.01, gravity=[0, 0, -50],
             action_lims=(-0.9, 0.9), polar_obs=True,
             scenario='quadrant', p_door_dropout=0.0,
             n_rooms=2, random_room_number=False,
             n_lidar_per_agent=0, visualize_lidar=False, compress_lidar_scale=None,
             n_boxes=2, box_size=0.5, box_only_z_rot=True,
             boxid_obs=False, boxsize_obs=True, pad_ramp_size=True, additional_obs={},
             # lock-box task
             task_type='order-return', lock_reward=5.0, unlock_penalty=5.0, shaped_reward_scale=0.5,
             return_threshold=0.5,
             # ramps
             n_ramps=0):

    scenario = args.scenario_name
    n_agents = args.num_agents
    task_type = args.task_type
    n_boxes = args.num_boxes
    floor_size = args.floor_size
    grid_size = args.grid_size
    fixed_door = args.fixed_door
    spawn_obs = args.spawn_obs
    horizon = args.env_horizon
    #assert n_agents==1, ("only 1 agents is supported, check the config.py.")
    
    grab_radius_multiplier = lock_grab_radius / box_size
    lock_radius_multiplier = lock_grab_radius / box_size
    env = Base(n_agents=n_agents, n_substeps=n_substeps,
               floor_size=floor_size,
               horizon=horizon, action_lims=action_lims, deterministic_mode=deterministic_mode,
               grid_size=grid_size)
    

    if scenario == 'randomwalls':
        env.add_module(RandomWalls(n_agents=n_agents, grid_size=grid_size, num_rooms=n_rooms,
                                   random_room_number=random_room_number,
                                   min_room_size=6, door_size=door_size,
                                   gen_door_obs=False))
        box_placement_fn = uniform_placement
        ramp_placement_fn = uniform_placement
        agent_placement_fn = uniform_placement if not fixed_agent_spawn else center_placement
    elif scenario == 'quadrant':
        env.add_module(WallScenarios(n_agents=n_agents, grid_size=grid_size, door_size=door_size,
                                     scenario=scenario, friction=other_friction,
                                     p_door_dropout=p_door_dropout,fixed_door=fixed_door))
        box_placement_fn = uniformOrset_placement
        # box_placement_fn = uniform_placement
        ramp_placement_fn = uniform_placement
        agent_placement_fn = quadrantOrset_placement if not fixed_agent_spawn else center_placement
        # agent_placement_fn = quadrant_placement if not fixed_agent_spawn else center_placement
    elif scenario == 'empty':
        # 初始态仍然放在quadrant范围内，去掉墙
        env.add_module(WallScenarios(n_agents=n_agents, grid_size=grid_size, door_size=2, scenario='empty'))
        q_size = int(0.5 * grid_size)
        env.metadata['quadrant_size'] = q_size
        box_placement_fn = uniformOrset_placement
        # box_placement_fn = uniform_placement
        ramp_placement_fn = uniform_placement
        agent_placement_fn = quadrantOrset_placement
        # agent_placement_fn = center_placement
    elif 'var_tri' in scenario:
        env.add_module(WallScenarios(n_agents=n_agents, grid_size=grid_size, door_size=door_size, scenario='var_tri'))
        ramp_placement_fn = [tri_placement(i % 3) for i in range(n_ramps)]
        agent_placement_fn = center_placement if fixed_agent_spawn else \
            (uniform_placement if 'uniform' in scenario else rotate_tri_placement)
        box_placement_fn = uniform_placement if 'uniform' in scenario else rotate_tri_placement
    else:
        raise ValueError(f"Scenario {scenario} not supported.")

    env.add_module(Agents(n_agents,
                          placement_fn=agent_placement_fn,
                          color=[np.array((66., 235., 244., 255.)) / 255] * n_agents,
                          friction=other_friction,
                          polar_obs=polar_obs))
    if np.max(n_boxes) > 0:
        env.add_module(Boxes(n_boxes=n_boxes, placement_fn=box_placement_fn,
                             friction=box_floor_friction, polar_obs=polar_obs,
                             n_elongated_boxes=0,
                             boxid_obs=boxid_obs,
                             box_only_z_rot=box_only_z_rot,
                             boxsize_obs=boxsize_obs))

    if n_ramps > 0:
        env.add_module(Ramps(n_ramps=n_ramps, placement_fn=ramp_placement_fn,
                             friction=other_friction, polar_obs=polar_obs,
                             pad_ramp_size=pad_ramp_size))

    if n_lidar_per_agent > 0 and visualize_lidar:
        env.add_module(LidarSites(n_agents=n_agents, n_lidar_per_agent=n_lidar_per_agent))

    if np.max(n_boxes) > 0 and grab_box:
        env.add_module(AgentManipulation())
    if box_floor_friction is not None:
        env.add_module(FloorAttributes(friction=box_floor_friction))
    env.add_module(WorldConstants(gravity=gravity))
    env.reset()
    if 'var_tri' in scenario or "randomwalls" in scenario or "empty" in scenario:
        keys_self = ['agent_qpos_qvel','current_step']
    else:
        keys_self = ['agent_qpos_qvel','current_step','vector_door_obs']

    if args.spawn_obs:
        keys_self += ['agent_spawn_obs']

    keys_mask_self = ['mask_aa_obs']
    keys_external = ['agent_qpos_qvel']
    keys_copy = ['you_lock', 'team_lock']
    keys_mask_external = []

    env = SplitMultiAgentActions(env)
    env = TeamMembership(env, np.zeros((n_agents,)))
    env = AgentAgentObsMask2D(env)
    env = DiscretizeActionWrapper(env, 'action_movement')
    env = NumpyArrayRewardWrapper(env)
    env = TimeWrapper(env, horizon)
    if np.max(n_boxes) > 0:
        env = AgentGeomObsMask2D(env, pos_obs_key='box_pos', mask_obs_key='mask_ab_obs',
                                 geom_idxs_obs_key='box_geom_idxs')
        keys_external += ['mask_ab_obs', 'box_obs']
        keys_mask_external.append('mask_ab_obs')
    if lock_box and np.max(n_boxes) > 0:
        env = LockObjWrapper(env, body_names=[f'moveable_box{i}' for i in range(n_boxes)],
                             agent_idx_allowed_to_lock=np.arange(n_agents),
                             lock_type=lock_type,
                             radius_multiplier=lock_radius_multiplier,
                             obj_in_game_metadata_keys=["curr_n_boxes"],
                             agent_allowed_to_lock_keys=None if lock_out_of_vision else ["mask_ab_obs"])

    if n_ramps > 0:
        env = AgentGeomObsMask2D(env, pos_obs_key='ramp_pos', mask_obs_key='mask_ar_obs',
                                 geom_idxs_obs_key='ramp_geom_idxs')
        env = LockObjWrapper(env, body_names=[f"ramp{i}:ramp" for i in range(n_ramps)],
                             agent_idx_allowed_to_lock=np.arange(n_agents),
                             lock_type=lock_type, ac_obs_prefix='ramp_',
                             radius_multiplier=lock_radius_multiplier,
                             agent_allowed_to_lock_keys=None if lock_out_of_vision else ["mask_ar_obs"])

        keys_external += ['ramp_obs']
        keys_mask_external += ['mask_ar_obs']
        keys_copy += ['ramp_you_lock', 'ramp_team_lock']

    if grab_box and np.max(n_boxes) > 0:
        body_names = ([f'moveable_box{i}' for i in range(n_boxes)] +
                      [f"ramp{i}:ramp" for i in range(n_ramps)])
        obj_in_game_meta_keys = ['curr_n_boxes'] + (['curr_n_ramps'] if n_ramps > 0 else [])
        env = GrabObjWrapper(env,
                             body_names=body_names,
                             radius_multiplier=grab_radius_multiplier,
                             grab_exclusive=grab_exclusive,
                             obj_in_game_metadata_keys=obj_in_game_meta_keys)

    if n_lidar_per_agent > 0:
        env = Lidar(env, n_lidar_per_agent=n_lidar_per_agent, visualize_lidar=visualize_lidar,
                    compress_lidar_scale=compress_lidar_scale)
        keys_copy += ['lidar']
        keys_external += ['lidar']

    env = AddConstantObservationsWrapper(env, new_obs=additional_obs)
    keys_external += list(additional_obs)
    keys_mask_external += [ob for ob in additional_obs if 'mask' in ob]

    #############################################
    # lock Box Task Reward
    ###
    env = LockObjectsTask(env, n_objs=n_boxes, task=task_type, fixed_order=True,
                          obj_lock_obs_key='obj_lock', obj_pos_obs_key='box_pos',
                          act_lock_key='action_glue', agent_pos_key='agent_pos',
                          lock_reward=lock_reward, unlock_penalty=unlock_penalty,
                          shaped_reward_scale=shaped_reward_scale,
                          return_threshold=return_threshold)
    ###
    #############################################

    env = SplitObservations(env, keys_self + keys_mask_self, keys_copy=keys_copy)
    env = SpoofEntityWrapper(env, n_boxes,
                             ['box_obs', 'you_lock', 'team_lock', 'obj_lock'],
                             ['mask_ab_obs'])
    keys_mask_external += ['mask_ab_obs_spoof']

    if n_agents < 2:
        env = SpoofEntityWrapper(env, 1, ['agent_qpos_qvel'], ['mask_aa_obs'])

    env = LockAllWrapper(env, remove_object_specific_lock=True)
    if not grab_out_of_vision and grab_box:
        # Can only pull if in vision
        mask_keys = ['mask_ab_obs'] + (['mask_ar_obs'] if n_ramps > 0 else [])
        env = MaskActionWrapper(env, action_key='action_pull', mask_keys=mask_keys)
    if not grab_selective and grab_box:
        env = GrabClosestWrapper(env)
    env = DiscardMujocoExceptionEpisodes(env,n_agents)
    if n_ramps > 0:
        env = ConcatenateObsWrapper(env, {'agent_qpos_qvel': ['agent_qpos_qvel'],
                                      'box_obs': ['box_obs', 'you_lock', 'team_lock', 'obj_lock'],
                                      'ramp_obs': ['ramp_obs', 'ramp_you_lock', 'ramp_team_lock',
                                                   'ramp_obj_lock']})
    else:
        env = ConcatenateObsWrapper(env, {'agent_qpos_qvel': ['agent_qpos_qvel'],
                                      'box_obs': ['box_obs', 'you_lock', 'team_lock', 'obj_lock']})
    
    env = SelectKeysWrapper(env, keys_self=keys_self,
                            keys_external=keys_external,
                            n_agents=n_agents,
                            n_boxes=n_boxes,
                            n_ramps=n_ramps,
                            keys_mask=keys_mask_self + keys_mask_external,
                            flatten=False)
    # init agent state
    # env = InitSetWrapper(env,n_agents=n_agents,n_boxes=n_boxes)
    return env
