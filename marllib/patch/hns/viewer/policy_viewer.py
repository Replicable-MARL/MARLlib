#!/usr/bin/env python
import time
import glfw
import numpy as np
from operator import itemgetter
from mujoco_py import const, MjViewer
from mujoco_worldgen.util.types import store_args
from marllib.patch.hns.ma_policy.util import listdict2dictnp
from functools import reduce
import pdb
import torch
import copy

def handle_dict_obs(keys, order_obs, mask_order_obs, dict_obs, num_agents, num_hiders):
    # obs = []
    # share_obs = []
    for i, key in enumerate(order_obs):
        if key in keys:             
            if mask_order_obs[i] == None:
                temp_share_obs = dict_obs[key].reshape(num_agents,-1).copy()
                temp_obs = temp_share_obs.copy()
            else:
                temp_share_obs = dict_obs[key].reshape(num_agents,-1).copy()
                temp_mask = dict_obs[mask_order_obs[i]].copy()
                temp_obs = dict_obs[key].copy()
                mins_temp_mask = ~temp_mask
                temp_obs[mins_temp_mask]=np.zeros((mins_temp_mask.sum(),temp_obs.shape[2]))                       
                temp_obs = temp_obs.reshape(num_agents,-1) 
            if i == 0:
                reshape_obs = temp_obs.copy()
                reshape_share_obs = temp_share_obs.copy()
            else:
                reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
    # obs.append(reshape_obs)
    # share_obs.append(reshape_share_obs)   
    # obs = np.array(obs)[:,num_hiders:]
    # share_obs = np.array(share_obs)[:,num_hiders:]
    obs = reshape_obs[num_hiders:]
    share_obs = reshape_share_obs[num_hiders:]
    return obs, share_obs

def splitobs(obs, keepdims=True):
    '''
        Split obs into list of single agent obs.
        Args:
            obs: dictionary of numpy arrays where first dim in each array is agent dim
    '''
    n_agents = obs[list(obs.keys())[0]].shape[0]
    return [{k: v[[i]] if keepdims else v[i] for k, v in obs.items()} for i in range(n_agents)]

class PolicyViewer(MjViewer):
    '''
    PolicyViewer runs a policy with an environment and optionally displays it.
        env - environment to run policy in
        policy - policy object to run
        display_window - if true, show the graphical viewer
        seed - environment seed to view
        duration - time in seconds to run the policy, run forever if duration=None
    '''
    @store_args
    def __init__(self, env, policies, display_window=True, seed=None, duration=None):
        if seed is None:
            self.seed = env.seed()[0]
        else:
            self.seed = seed
            env.seed(seed)
        self.total_rew = 0.0
        self.ob = env.reset()
        for policy in self.policies:
            policy.reset()
        assert env.metadata['n_actors'] % len(policies) == 0
        if hasattr(env, "reset_goal"):
            self.goal = env.reset_goal()
        super().__init__(self.env.unwrapped.sim)
        # TO DO: remove circular dependency on viewer object. It looks fishy.
        self.env.unwrapped.viewer = self
        if self.render and self.display_window:
            self.env.render()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        # Trigger on keyup only:
        if action != glfw.RELEASE:
            return
        # Increment experiment seed
        if key == glfw.KEY_N:
            self.reset_increment()
        # Decrement experiment trial
        elif key == glfw.KEY_P:
            print("Pressed P")
            self.seed = max(self.seed - 1, 0)
            self.env.seed(self.seed)
            self.ob = self.env.reset()
            for policy in self.policies:
                policy.reset()
            if hasattr(self.env, "reset_goal"):
                self.goal = self.env.reset_goal()
            self.update_sim(self.env.unwrapped.sim)

    def run(self):
        if self.duration is not None:
            self.end_time = time.time() + self.duration
        self.total_rew_avg = 0.0
        self.n_episodes = 0
        while self.duration is None or time.time() < self.end_time:
            if len(self.policies) == 1:
                action, _ = self.policies[0].act(self.ob)
            else:
                self.ob = splitobs(self.ob, keepdims=False)
                ob_policy_idx = np.split(np.arange(len(self.ob)), len(self.policies))
                actions = []
                for i, policy in enumerate(self.policies):
                    inp = itemgetter(*ob_policy_idx[i])(self.ob)
                    inp = listdict2dictnp([inp] if ob_policy_idx[i].shape[0] == 1 else inp)
                    ac, info = policy.act(inp)
                    actions.append(ac)
                action = listdict2dictnp(actions, keepdims=True)
            
            self.ob, rew, done, env_info = self.env.step(action)
            self.total_rew += rew

            if done or env_info.get('discard_episode', False):
                self.reset_increment()

            if self.display_window:
                self.add_overlay(const.GRID_TOPRIGHT, "Reset env; (current seed: {})".format(self.seed), "N - next / P - previous ")
                self.add_overlay(const.GRID_TOPRIGHT, "Reward", str(self.total_rew))
                if hasattr(self.env.unwrapped, "viewer_stats"):
                    for k, v in self.env.unwrapped.viewer_stats.items():
                        self.add_overlay(const.GRID_TOPRIGHT, k, str(v))

                self.env.render()

    def reset_increment(self):
        self.total_rew_avg = (self.n_episodes * self.total_rew_avg + self.total_rew) / (self.n_episodes + 1)
        self.n_episodes += 1
        print(f"Reward: {self.total_rew} (rolling average: {self.total_rew_avg})")
        self.total_rew = 0.0
        self.seed += 1
        self.env.seed(self.seed)
        self.ob = self.env.reset()
        for policy in self.policies:
            policy.reset()
        if hasattr(self.env, "reset_goal"):
            self.goal = self.env.reset_goal()
        self.update_sim(self.env.unwrapped.sim)

class PolicyViewer_hs_single(MjViewer):
    '''
    PolicyViewer runs a policy with an environment and optionally displays it.
        env - environment to run policy in
        policy - policy object to run
        display_window - if true, show the graphical viewer
        seed - environment seed to view
        duration - time in seconds to run the policy, run forever if duration=None
    '''
    @store_args
    def __init__(self, all_args, env, policies, display_window=True, seed=None, duration=None):
        if seed is None:
            self.seed = env.seed()[0]
        else:
            self.seed = seed
            env.seed(seed)
        self.total_rew = 0.0
        self.dict_obs = env.reset()
        #for policy in self.policies:
        #    policy.reset()
        assert env.metadata['n_actors'] % len(policies) == 0
        if hasattr(env, "reset_goal"):
            self.goal = env.reset_goal()
        super().__init__(self.env.unwrapped.sim)
        # TO DO: remove circular dependency on viewer object. It looks fishy.
        self.env.unwrapped.viewer = self
        if self.render and self.display_window:
            self.env.render()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        # Trigger on keyup only:
        if action != glfw.RELEASE:
            return
        # Increment experiment seed
        if key == glfw.KEY_N:
            self.reset_increment()
        # Decrement experiment trial
        elif key == glfw.KEY_P:
            print("Pressed P")
            self.seed = max(self.seed - 1, 0)
            self.env.seed(self.seed)
            self.ob = self.env.reset()
            for policy in self.policies:
                policy.reset()
            if hasattr(self.env, "reset_goal"):
                self.goal = self.env.reset_goal()
            self.update_sim(self.env.unwrapped.sim)

    def run(self):
        self.action_movement_dim = []
        '''
        self.order_obs = ['agent_qpos_qvel','box_obs','ramp_obs','food_obs','observation_self']    
        self.mask_order_obs = ['mask_aa_obs','mask_ab_obs','mask_ar_obs','mask_af_obs',None]
        '''
        # self.order_obs = ['agent_qpos_qvel', 'box_obs','ramp_obs','construction_site_obs','vector_door_obs', 'observation_self']    
        # self.mask_order_obs = ['mask_aa_obs', 'mask_ab_obs','mask_ar_obs',None,None,None]
        self.order_obs = ['agent_qpos_qvel', 'box_obs','ramp_obs','construction_site_obs', 'observation_self']    
        self.mask_order_obs = [None,None,None,None,None]
        self.keys = self.env.observation_space.spaces.keys()
   
        self.num_agents = 2
        self.num_hiders = 1
        self.num_seekers = 1
        for agent_id in range(self.num_agents):
            # deal with dict action space
            action_movement = self.env.action_space['action_movement'][agent_id].nvec
            self.action_movement_dim.append(len(action_movement))
        self.masks = np.ones((1, self.num_agents, 1)).astype(np.float32)
        if self.duration is not None:
            self.end_time = time.time() + self.duration
        self.total_rew_avg = 0.0
        self.n_episodes = 0
        self.obs = []
        self.share_obs = []
        reshape_obs, reshape_share_obs = handle_dict_obs(self.keys, self.order_obs, self.mask_order_obs, self.dict_obs, self.num_agents, self.num_hiders)  
        self.obs.append(reshape_obs)
        self.share_obs.append(reshape_share_obs)   
        self.obs = np.array(self.obs).astype(np.float32)
        self.share_obs = np.array(self.share_obs).astype(np.float32)
        self.recurrent_hidden_states = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        self.recurrent_hidden_states_critic = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        while self.duration is None or time.time() < self.end_time:
            values = []
            actions= []
            recurrent_hidden_statess = []
            recurrent_hidden_statess_critic = []
            with torch.no_grad():                
                for agent_id in range(self.num_seekers):
                    self.policies[0].eval()
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = self.policies[0].act(agent_id,
                    torch.tensor(self.share_obs[:,agent_id,:]), 
                    torch.tensor(self.obs[:,agent_id,:]), 
                    torch.tensor(self.recurrent_hidden_states[:,agent_id,:]), 
                    torch.tensor(self.recurrent_hidden_states_critic[:,agent_id,:]),
                    torch.tensor(self.masks[:,agent_id,:]))
                    values.append(value.detach().cpu().numpy())
                    actions.append(action.detach().cpu().numpy())
                    recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
                    recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())
            
            # rearrange action        
            action_movement = []
            action_pull = []
            action_glueall = []
            for k in range(self.num_hiders):
                #action_movement.append(np.random.randint(11, size=3))  #hider随机游走
                action_movement.append(np.array([5,5,5]))   #hider静止不动
                action_pull.append(0)
                action_glueall.append(0)
            for k in range(self.num_seekers):
                action_movement.append(actions[k][0][:3])
                action_pull.append(np.int(actions[k][0][3]))
                action_glueall.append(np.int(actions[k][0][4]))
            action_movement = np.stack(action_movement, axis = 0)
            action_glueall = np.stack(action_glueall, axis = 0)
            action_pull = np.stack(action_pull, axis = 0)                        
            one_env_action = {'action_movement': action_movement, 'action_pull': action_pull, 'action_glueall': action_glueall}

            self.dict_obs, rew, done, env_info = self.env.step(one_env_action)
            self.total_rew += rew
            self.obs = []
            self.share_obs = []   
            reshape_obs, reshape_share_obs = handle_dict_obs(self.keys, self.order_obs, self.mask_order_obs, self.dict_obs, self.num_agents, self.num_hiders)               
            self.obs.append(reshape_obs)
            self.share_obs.append(reshape_share_obs)   
            self.obs = np.array(self.obs).astype(np.float32)
            self.share_obs = np.array(self.share_obs).astype(np.float32)
            self.recurrent_hidden_states = np.array(recurrent_hidden_statess).transpose(1,0,2)
            self.recurrent_hidden_states_critic = np.array(recurrent_hidden_statess_critic).transpose(1,0,2)
            if done or env_info.get('discard_episode', False):
                self.reset_increment()

            if self.display_window:
                self.add_overlay(const.GRID_TOPRIGHT, "Reset env; (current seed: {})".format(self.seed), "N - next / P - previous ")
                self.add_overlay(const.GRID_TOPRIGHT, "Reward", str(self.total_rew))
                if hasattr(self.env.unwrapped, "viewer_stats"):
                    for k, v in self.env.unwrapped.viewer_stats.items():
                        self.add_overlay(const.GRID_TOPRIGHT, k, str(v))

                self.env.render()

    def reset_increment(self):
        self.total_rew_avg = (self.n_episodes * self.total_rew_avg + self.total_rew) / (self.n_episodes + 1)
        self.n_episodes += 1
        print(f"Reward: {self.total_rew} (rolling average: {self.total_rew_avg})")
        self.total_rew = 0.0
        self.seed += 1
        self.env.seed(self.seed)
        self.dict_obs = self.env.reset()
        self.obs = []
        self.share_obs = []   
        reshape_obs, reshape_share_obs = handle_dict_obs(self.keys, self.order_obs, self.mask_order_obs, self.dict_obs, self.num_agents, self.num_hiders)             
        self.obs.append(reshape_obs)
        self.share_obs.append(reshape_share_obs)   
        self.obs = np.array(self.obs).astype(np.float32)
        self.share_obs = np.array(self.share_obs).astype(np.float32) 
        self.recurrent_hidden_states = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        self.recurrent_hidden_states_critic = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        #for policy in self.policies:
        #    policy.reset()
        if hasattr(self.env, "reset_goal"):
            self.goal = self.env.reset_goal()
        self.update_sim(self.env.unwrapped.sim)

class PolicyViewer_bl(MjViewer):
    '''
    PolicyViewer runs a policy with an environment and optionally displays it.
        env - environment to run policy in
        policy - policy object to run
        display_window - if true, show the graphical viewer
        seed - environment seed to view
        duration - time in seconds to run the policy, run forever if duration=None
    '''
    @store_args
    def __init__(self, args, env, policies, display_window=True, seed=None, duration=None):
        if seed is None:
            self.seed = env.seed()[0]
        else:
            self.seed = seed
            env.seed(seed)
        self.args = args
        self.num_agents = args.num_agents
        self.total_rew = 0.0
        self.dict_obs = env.reset()
        self.eval_num = 10
        self.eval_episode = 0
        self.success_rate_sum = 0
        self.step = 0
        self.H = 5
        #for policy in self.policies:
        #    policy.reset()
        #assert env.metadata['n_actors'] % len(policies) == 0
        if hasattr(env, "reset_goal"):
            self.goal = env.reset_goal()
        super().__init__(self.env.unwrapped.sim)
        # TO DO: remove circular dependency on viewer object. It looks fishy.
        self.env.unwrapped.viewer = self
        if self.render and self.display_window:
            self.env.render()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        # Trigger on keyup only:
        if action != glfw.RELEASE:
            return
        # Increment experiment seed
        if key == glfw.KEY_N:
            self.reset_increment()
        # Decrement experiment trial
        elif key == glfw.KEY_P:
            print("Pressed P")
            self.seed = max(self.seed - 1, 0)
            self.env.seed(self.seed)
            self.ob = self.env.reset()
            for policy in self.policies:
                policy.reset()
            if hasattr(self.env, "reset_goal"):
                self.goal = self.env.reset_goal()
            self.update_sim(self.env.unwrapped.sim)

    def run(self):
        self.action_movement_dim = []

        self.order_obs = ['agent_qpos_qvel', 'box_obs', 'ramp_obs', 'construction_site_obs', 'observation_self']    
        self.mask_order_obs = [None, None, None, None, None]

        for agent_id in range(self.num_agents):
            # deal with dict action space
            action_movement = self.env.action_space['action_movement'][agent_id].nvec
            self.action_movement_dim.append(len(action_movement))

        # generate the obs space
        obs_shape = []
        obs_dim = 0
        for key in self.order_obs:
            if key in self.env.observation_space.spaces.keys():
                space = list(self.env.observation_space[key].shape)
                if len(space)<2:  
                    space.insert(0,1)        
                obs_shape.append(space)
                obs_dim += reduce(lambda x,y:x*y,space)
        obs_shape.insert(0,obs_dim)
        split_shape = obs_shape[1:]
        self.policies[0].base.obs_shape = obs_shape
        self.policies[0].base.encoder_actor.embedding.split_shape = split_shape
        self.policies[0].base.encoder_critic.embedding.split_shape = split_shape

        self.masks = np.ones((1, self.num_agents, 1)).astype(np.float32)
        if self.duration is not None:
            self.end_time = time.time() + self.duration
        self.total_rew_avg = 0.0
        self.n_episodes = 0
        self.obs = []
        self.share_obs = []   
        for i, key in enumerate(self.order_obs):
            if key in self.env.observation_space.spaces.keys():             
                if self.mask_order_obs[i] == None:          
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_obs = temp_share_obs.copy()
                else:
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_mask = self.dict_obs[self.mask_order_obs[i]].copy()
                    temp_obs = self.dict_obs[key].copy()
                    mins_temp_mask = ~temp_mask
                    temp_obs[mins_temp_mask]=np.zeros(((mins_temp_mask).sum(),temp_obs.shape[2]))                       
                    temp_obs = temp_obs.reshape(self.num_agents,-1) 
                if i == 0:
                    reshape_obs = temp_obs.copy()
                    reshape_share_obs = temp_share_obs.copy()
                else:
                    reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                    reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
        self.obs.append(reshape_obs)
        self.share_obs.append(reshape_share_obs)   
        self.obs = np.array(self.obs).astype(np.float32)
        self.share_obs = np.array(self.share_obs).astype(np.float32) 
        self.recurrent_hidden_states = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        self.recurrent_hidden_states_critic = np.zeros((1, self.num_agents, 64)).astype(np.float32)

        self.test_lock_rate = np.zeros(self.args.episode_length)
        self.test_return_rate = np.zeros(self.args.episode_length)
        self.test_success_rate = np.zeros(self.args.episode_length)

        while (self.duration is None or time.time() < self.end_time) and self.eval_episode < self.eval_num:
            values = []
            actions= []
            recurrent_hidden_statess = []
            recurrent_hidden_statess_critic = []
            with torch.no_grad():                
                for agent_id in range(self.num_agents):
                    self.policies[0].eval()
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = self.policies[0].act(agent_id,
                    torch.tensor(self.share_obs[:,agent_id,:]), 
                    torch.tensor(self.obs[:,agent_id,:]), 
                    torch.tensor(self.recurrent_hidden_states[:,agent_id,:]), 
                    torch.tensor(self.recurrent_hidden_states_critic[:,agent_id,:]),
                    torch.tensor(self.masks[:,agent_id,:]))
                    values.append(value.detach().cpu().numpy())
                    actions.append(action.detach().cpu().numpy())
                    recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
                    recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())

            action_movement = []
            action_pull = []
            action_glueall = []
            for agent_id in range(self.num_agents):
                action_movement.append(actions[agent_id][0][:self.action_movement_dim[agent_id]])
                action_glueall.append(int(actions[agent_id][0][self.action_movement_dim[agent_id]]))
                if 'action_pull' in self.env.action_space.spaces.keys():
                    action_pull.append(int(actions[agent_id][0][-1]))
            action_movement = np.stack(action_movement, axis = 0)
            action_glueall = np.stack(action_glueall, axis = 0)
            if 'action_pull' in self.env.action_space.spaces.keys():
                action_pull = np.stack(action_pull, axis = 0)                             
            one_env_action = {'action_movement': action_movement, 'action_pull': action_pull, 'action_glueall': action_glueall}
            self.dict_obs, rew, done, env_info = self.env.step(one_env_action)
            self.step += 1

            #READ INFO
            self.test_lock_rate[self.step] = env_info['lock_rate']
            self.test_return_rate[self.step] = env_info['return_rate']
            if env_info['lock_rate'] == 1:
                self.test_success_rate[self.step] = env_info['return_rate']
            else:
                self.test_success_rate[self.step] = 0
            # print("Step %d Lock Rate"%self.step, self.test_lock_rate[self.step])
            # print("Step %d Return Rate"%self.step, self.test_return_rate[self.step])
            # print("Step %d Success Rate"%self.step, self.test_success_rate[self.step])
            

            #print(self.dict_obs['box_obs'][0][0])
            self.total_rew += rew
            self.is_lock = self.test_lock_rate[self.step]
            self.is_return = self.test_return_rate[self.step]
            self.obs = []
            self.share_obs = []   
            for i, key in enumerate(self.order_obs):
                if key in self.env.observation_space.spaces.keys():             
                    if self.mask_order_obs[i] == None:
                        temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                        temp_obs = temp_share_obs.copy()
                    else:
                        temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                        temp_mask = self.dict_obs[self.mask_order_obs[i]].copy()
                        temp_obs = self.dict_obs[key].copy()
                        mins_temp_mask = ~temp_mask
                        temp_obs[mins_temp_mask]=np.zeros(((mins_temp_mask).sum(),temp_obs.shape[2]))                       
                        temp_obs = temp_obs.reshape(self.num_agents,-1) 
                    if i == 0:
                        reshape_obs = temp_obs.copy()
                        reshape_share_obs = temp_share_obs.copy()
                    else:
                        reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                        reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
            self.obs.append(reshape_obs)
            self.share_obs.append(reshape_share_obs)   
            self.obs = np.array(self.obs).astype(np.float32)
            self.share_obs = np.array(self.share_obs).astype(np.float32)
            self.recurrent_hidden_states = np.array(recurrent_hidden_statess).transpose(1,0,2)
            self.recurrent_hidden_states_critic = np.array(recurrent_hidden_statess_critic).transpose(1,0,2)
            if done or env_info.get('discard_episode', False) or self.step >= self.args.episode_length - 1:
                self.eval_episode += 1
                self.success_rate_sum += np.mean(self.test_success_rate[-self.H:])
                print("Test Episode %d/%d Success Rate:"%(self.eval_episode, self.eval_num), np.mean(self.test_success_rate[-self.H:]))
                self.reset_increment()

            if self.display_window:
                self.add_overlay(const.GRID_TOPRIGHT, "Reset env; (current seed: {})".format(self.seed), "N - next / P - previous ")
                self.add_overlay(const.GRID_TOPRIGHT, "Reward", str(self.total_rew))
                self.add_overlay(const.GRID_TOPRIGHT, "Lock", str(self.is_lock))
                self.add_overlay(const.GRID_TOPRIGHT, "Return", str(self.is_return))
                if hasattr(self.env.unwrapped, "viewer_stats"):
                    for k, v in self.env.unwrapped.viewer_stats.items():
                        self.add_overlay(const.GRID_TOPRIGHT, k, str(v))

                self.env.render()
        
        if self.eval_episode == self.eval_num:
            print("Mean Success Rate:", self.success_rate_sum / self.eval_num)

    def reset_increment(self):
        self.total_rew_avg = (self.n_episodes * self.total_rew_avg + self.total_rew) / (self.n_episodes + 1)
        self.n_episodes += 1
        print(f"Reward: {self.total_rew} (rolling average: {self.total_rew_avg})")
        self.total_rew = 0.0
        self.seed += 1
        self.env.seed(self.seed)
        
        self.dict_obs = self.env.reset()
        self.obs = []
        self.share_obs = []

        # reset the buffer
        self.test_lock_rate = np.zeros(self.args.episode_length)
        self.test_return_rate = np.zeros(self.args.episode_length)
        self.test_success_rate = np.zeros(self.args.episode_length)
        self.step = 0

        for i, key in enumerate(self.order_obs):
            if key in self.env.observation_space.spaces.keys():             
                if self.mask_order_obs[i] == None:
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_obs = temp_share_obs.copy()
                else:
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_mask = self.dict_obs[self.mask_order_obs[i]].copy()
                    temp_obs = self.dict_obs[key].copy()
                    mins_temp_mask = ~temp_mask
                    temp_obs[mins_temp_mask]=np.zeros(((mins_temp_mask).sum(),temp_obs.shape[2]))                       
                    temp_obs = temp_obs.reshape(self.num_agents,-1) 
                if i == 0:
                    reshape_obs = temp_obs.copy()
                    reshape_share_obs = temp_share_obs.copy()
                else:
                    reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                    reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
        self.obs.append(reshape_obs)
        self.share_obs.append(reshape_share_obs)   
        self.obs = np.array(self.obs).astype(np.float32)
        self.share_obs = np.array(self.share_obs).astype(np.float32) 
        self.recurrent_hidden_states = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        self.recurrent_hidden_states_critic = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        #for policy in self.policies:
        #    policy.reset()
        if hasattr(self.env, "reset_goal"):
            self.goal = self.env.reset_goal()
        self.update_sim(self.env.unwrapped.sim)

class PolicyViewer_bl_good_case(MjViewer):
    '''
    PolicyViewer runs a policy with an environment and optionally displays it.
        env - environment to run policy in
        policy - policy object to run
        display_window - if true, show the graphical viewer
        seed - environment seed to view
        duration - time in seconds to run the policy, run forever if duration=None
    '''
    @store_args
    def __init__(self, args, env, policies, display_window=True, seed=None, duration=None):
        if seed is None:
            self.seed = env.seed()[0]
        else:
            self.seed = seed
            env.seed(seed)
        self.args = args
        self.num_agents = args.num_agents
        self.total_rew = 0.0
        
        # init starts
        self.eval_num = 1
        self.eval_episode = 0
        self.success_rate_sum = 0
        self.step = 0
        self.H = 5
        buffer_length = 2000
        boundary = args.grid_size-2
        boundary_quadrant = [round(args.grid_size / 2), args.grid_size-3, 1, round(args.grid_size/2)-3]
        start_boundary = [round(args.grid_size / 2), args.grid_size-3, 1, round(args.grid_size/2)-3] # x1,x2,y1,y2 qudrant set
        last_node = node_buffer(args.num_agents, args.num_boxes, buffer_length,
                        archive_initial_length=args.n_rollout_threads,
                        reproduction_num=160,
                        max_step=1,
                        start_boundary=start_boundary,
                        boundary=boundary,
                        boundary_quadrant=boundary_quadrant)
        #self.starts = last_node.produce_good_case(self.eval_num, start_boundary, args.num_agents, args.num_boxes)
        self.starts = [[np.array([16,  4]), np.array([21,  2]), np.array([22,  2]), np.array([16,  4])]]
        print("[starts]", self.starts[0])
        self.dict_obs = env.reset(self.starts[0])
        #for policy in self.policies:
        #    policy.reset()
        #assert env.metadata['n_actors'] % len(policies) == 0
        if hasattr(env, "reset_goal"):
            self.goal = env.reset_goal()
        super().__init__(self.env.unwrapped.sim)
        # TO DO: remove circular dependency on viewer object. It looks fishy.
        self.env.unwrapped.viewer = self
        if self.render and self.display_window:
            self.env.render()
        


    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        # Trigger on keyup only:
        if action != glfw.RELEASE:
            return
        # Increment experiment seed
        if key == glfw.KEY_N:
            self.reset_increment()
        # Decrement experiment trial
        elif key == glfw.KEY_P:
            print("Pressed P")
            self.seed = max(self.seed - 1, 0)
            self.env.seed(self.seed)
            self.ob = self.env.reset()
            for policy in self.policies:
                policy.reset()
            if hasattr(self.env, "reset_goal"):
                self.goal = self.env.reset_goal()
            self.update_sim(self.env.unwrapped.sim)

    def run(self):
        self.action_movement_dim = []

        self.order_obs = ['agent_qpos_qvel', 'box_obs', 'ramp_obs', 'construction_site_obs', 'observation_self']    
        self.mask_order_obs = [None, None, None, None, None]

        for agent_id in range(self.num_agents):
            # deal with dict action space
            action_movement = self.env.action_space['action_movement'][agent_id].nvec
            self.action_movement_dim.append(len(action_movement))

        # generate the obs space
        obs_shape = []
        obs_dim = 0
        for key in self.order_obs:
            if key in self.env.observation_space.spaces.keys():
                space = list(self.env.observation_space[key].shape)
                if len(space)<2:  
                    space.insert(0,1)        
                obs_shape.append(space)
                obs_dim += reduce(lambda x,y:x*y,space)
        obs_shape.insert(0,obs_dim)
        split_shape = obs_shape[1:]
        self.policies[0].base.obs_shape = obs_shape
        self.policies[0].base.encoder_actor.embedding.split_shape = split_shape
        self.policies[0].base.encoder_critic.embedding.split_shape = split_shape

        self.masks = np.ones((1, self.num_agents, 1)).astype(np.float32)
        if self.duration is not None:
            self.end_time = time.time() + self.duration
        self.total_rew_avg = 0.0
        self.n_episodes = 0
        self.obs = []
        self.share_obs = []   
        for i, key in enumerate(self.order_obs):
            if key in self.env.observation_space.spaces.keys():             
                if self.mask_order_obs[i] == None:          
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_obs = temp_share_obs.copy()
                else:
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_mask = self.dict_obs[self.mask_order_obs[i]].copy()
                    temp_obs = self.dict_obs[key].copy()
                    mins_temp_mask = ~temp_mask
                    temp_obs[mins_temp_mask]=np.zeros(((mins_temp_mask).sum(),temp_obs.shape[2]))                       
                    temp_obs = temp_obs.reshape(self.num_agents,-1) 
                if i == 0:
                    reshape_obs = temp_obs.copy()
                    reshape_share_obs = temp_share_obs.copy()
                else:
                    reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                    reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
        self.obs.append(reshape_obs)
        self.share_obs.append(reshape_share_obs)   
        self.obs = np.array(self.obs).astype(np.float32)
        self.share_obs = np.array(self.share_obs).astype(np.float32) 
        self.recurrent_hidden_states = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        self.recurrent_hidden_states_critic = np.zeros((1, self.num_agents, 64)).astype(np.float32)

        self.test_lock_rate = np.zeros(self.args.episode_length)
        self.test_return_rate = np.zeros(self.args.episode_length)
        self.test_success_rate = np.zeros(self.args.episode_length)

        while self.duration is None or time.time() < self.end_time or self.eval_episode <= self.eval_num:
            values = []
            actions= []
            recurrent_hidden_statess = []
            recurrent_hidden_statess_critic = []
            with torch.no_grad():                
                for agent_id in range(self.num_agents):
                    self.policies[0].eval()
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = self.policies[0].act(agent_id,
                    torch.tensor(self.share_obs[:,agent_id,:]), 
                    torch.tensor(self.obs[:,agent_id,:]), 
                    torch.tensor(self.recurrent_hidden_states[:,agent_id,:]), 
                    torch.tensor(self.recurrent_hidden_states_critic[:,agent_id,:]),
                    torch.tensor(self.masks[:,agent_id,:]))
                    values.append(value.detach().cpu().numpy())
                    actions.append(action.detach().cpu().numpy())
                    recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
                    recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())

            action_movement = []
            action_pull = []
            action_glueall = []
            for agent_id in range(self.num_agents):
                action_movement.append(actions[agent_id][0][:self.action_movement_dim[agent_id]])
                action_glueall.append(int(actions[agent_id][0][self.action_movement_dim[agent_id]]))
                if 'action_pull' in self.env.action_space.spaces.keys():
                    action_pull.append(int(actions[agent_id][0][-1]))
            action_movement = np.stack(action_movement, axis = 0)
            action_glueall = np.stack(action_glueall, axis = 0)
            if 'action_pull' in self.env.action_space.spaces.keys():
                action_pull = np.stack(action_pull, axis = 0)                             
            one_env_action = {'action_movement': action_movement, 'action_pull': action_pull, 'action_glueall': action_glueall}
            self.dict_obs, rew, done, env_info = self.env.step(one_env_action)
            self.step += 1
            #READ INFO
            self.test_lock_rate[self.step] = env_info['lock_rate']
            self.test_return_rate[self.step] = env_info['return_rate']
            if env_info['lock_rate'] == 1:
                self.test_success_rate[self.step] = env_info['return_rate']
            else:
                self.test_success_rate[self.step] = 0
            #print(self.dict_obs['box_obs'][0][0])
            self.total_rew += rew
            self.obs = []
            self.share_obs = []   
            for i, key in enumerate(self.order_obs):
                if key in self.env.observation_space.spaces.keys():             
                    if self.mask_order_obs[i] == None:
                        temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                        temp_obs = temp_share_obs.copy()
                    else:
                        temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                        temp_mask = self.dict_obs[self.mask_order_obs[i]].copy()
                        temp_obs = self.dict_obs[key].copy()
                        mins_temp_mask = ~temp_mask
                        temp_obs[mins_temp_mask]=np.zeros(((mins_temp_mask).sum(),temp_obs.shape[2]))                       
                        temp_obs = temp_obs.reshape(self.num_agents,-1) 
                    if i == 0:
                        reshape_obs = temp_obs.copy()
                        reshape_share_obs = temp_share_obs.copy()
                    else:
                        reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                        reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
            self.obs.append(reshape_obs)
            self.share_obs.append(reshape_share_obs)   
            self.obs = np.array(self.obs).astype(np.float32)
            self.share_obs = np.array(self.share_obs).astype(np.float32)
            self.recurrent_hidden_states = np.array(recurrent_hidden_statess).transpose(1,0,2)
            self.recurrent_hidden_states_critic = np.array(recurrent_hidden_statess_critic).transpose(1,0,2)
            if done or env_info.get('discard_episode', False) or self.step >= self.args.episode_length - 1:
                self.eval_episode += 1
                self.success_rate_sum += np.mean(self.test_success_rate[-self.H:])
                print("Test Episode %d/%d Success Rate:"%(self.eval_episode, self.eval_num), np.mean(self.test_success_rate[-self.H:]))
                if self.eval_episode == self.eval_num:
                    break
                self.reset_increment()

            if self.display_window:
                self.add_overlay(const.GRID_TOPRIGHT, "Reset env; (current seed: {})".format(self.seed), "N - next / P - previous ")
                self.add_overlay(const.GRID_TOPRIGHT, "Reward", str(self.total_rew))
                if hasattr(self.env.unwrapped, "viewer_stats"):
                    for k, v in self.env.unwrapped.viewer_stats.items():
                        self.add_overlay(const.GRID_TOPRIGHT, k, str(v))

                self.env.render()

        if self.eval_episode == self.eval_num:
            print("Mean Success Rate:", self.success_rate_sum / self.eval_num)
            
            

    def reset_increment(self):
        self.total_rew_avg = (self.n_episodes * self.total_rew_avg + self.total_rew) / (self.n_episodes + 1)
        self.n_episodes += 1
        print(f"Reward: {self.total_rew} (rolling average: {self.total_rew_avg})")
        self.total_rew = 0.0
        self.seed += 1
        self.env.seed(self.seed)
        print("[starts]", self.starts[self.eval_episode])
        self.dict_obs = self.env.reset(self.starts[self.eval_episode])
        self.obs = []
        self.share_obs = [] 

        # reset the buffer
        self.test_lock_rate = np.zeros(self.args.episode_length)
        self.test_return_rate = np.zeros(self.args.episode_length)
        self.test_success_rate = np.zeros(self.args.episode_length)
        self.step = 0
  
        for i, key in enumerate(self.order_obs):
            if key in self.env.observation_space.spaces.keys():             
                if self.mask_order_obs[i] == None:
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_obs = temp_share_obs.copy()
                else:
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_mask = self.dict_obs[self.mask_order_obs[i]].copy()
                    temp_obs = self.dict_obs[key].copy()
                    mins_temp_mask = ~temp_mask
                    temp_obs[mins_temp_mask]=np.zeros(((mins_temp_mask).sum(),temp_obs.shape[2]))                       
                    temp_obs = temp_obs.reshape(self.num_agents,-1) 
                if i == 0:
                    reshape_obs = temp_obs.copy()
                    reshape_share_obs = temp_share_obs.copy()
                else:
                    reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                    reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
        self.obs.append(reshape_obs)
        self.share_obs.append(reshape_share_obs)   
        self.obs = np.array(self.obs).astype(np.float32)
        self.share_obs = np.array(self.share_obs).astype(np.float32) 
        self.recurrent_hidden_states = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        self.recurrent_hidden_states_critic = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        #for policy in self.policies:
        #    policy.reset()
        if hasattr(self.env, "reset_goal"):
            self.goal = self.env.reset_goal()
        self.update_sim(self.env.unwrapped.sim)

class PolicyViewer_sc(MjViewer):
    '''
    PolicyViewer runs a policy with an environment and optionally displays it.
        env - environment to run policy in
        policy - policy object to run
        display_window - if true, show the graphical viewer
        seed - environment seed to view
        duration - time in seconds to run the policy, run forever if duration=None
    '''
    @store_args
    def __init__(self, env, policies, display_window=True, seed=None, duration=None):
        if seed is None:
            self.seed = env.seed()[0]
        else:
            self.seed = seed
            env.seed(seed)
        self.total_rew = 0.0
        self.dict_obs = env.reset()
        #for policy in self.policies:
        #    policy.reset()
        #assert env.metadata['n_actors'] % len(policies) == 0
        if hasattr(env, "reset_goal"):
            self.goal = env.reset_goal()
        super().__init__(self.env.unwrapped.sim)
        # TO DO: remove circular dependency on viewer object. It looks fishy.
        self.env.unwrapped.viewer = self
        if self.render and self.display_window:
            self.env.render()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        # Trigger on keyup only:
        if action != glfw.RELEASE:
            return
        # Increment experiment seed
        if key == glfw.KEY_N:
            self.reset_increment()
        # Decrement experiment trial
        elif key == glfw.KEY_P:
            print("Pressed P")
            self.seed = max(self.seed - 1, 0)
            self.env.seed(self.seed)
            self.ob = self.env.reset()
            for policy in self.policies:
                policy.reset()
            if hasattr(self.env, "reset_goal"):
                self.goal = self.env.reset_goal()
            self.update_sim(self.env.unwrapped.sim)

    def run(self):
        self.action_movement_dim = []
        '''
        self.order_obs = ['agent_qpos_qvel','box_obs','ramp_obs','food_obs','observation_self']    
        self.mask_order_obs = ['mask_aa_obs','mask_ab_obs','mask_ar_obs','mask_af_obs',None]
        '''
        self.order_obs = ['box_obs','ramp_obs','construction_site_obs','vector_door_obs', 'observation_self']    
        self.mask_order_obs = ['mask_ab_obs','mask_ar_obs',None,None,None]
        self.num_agents = 1
        for agent_id in range(self.num_agents):
            # deal with dict action space
            action_movement = self.env.action_space['action_movement'][agent_id].nvec
            self.action_movement_dim.append(len(action_movement))
        self.masks = np.ones((1, self.num_agents, 1)).astype(np.float32)
        if self.duration is not None:
            self.end_time = time.time() + self.duration
        self.total_rew_avg = 0.0
        self.n_episodes = 0
        self.obs = []
        self.share_obs = []   
        print(self.dict_obs)
        for i, key in enumerate(self.order_obs):
            if key in self.env.observation_space.spaces.keys():             
                if self.mask_order_obs[i] == None:          
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_obs = temp_share_obs.copy()
                else:
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_mask = self.dict_obs[self.mask_order_obs[i]].copy()
                    temp_obs = self.dict_obs[key].copy()
                    mins_temp_mask = ~temp_mask
                    temp_obs[mins_temp_mask]=np.zeros(((mins_temp_mask).sum(),temp_obs.shape[2]))                       
                    temp_obs = temp_obs.reshape(self.num_agents,-1) 
                if i == 0:
                    reshape_obs = temp_obs.copy()
                    reshape_share_obs = temp_share_obs.copy()
                else:
                    reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                    reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
        self.obs.append(reshape_obs)
        self.share_obs.append(reshape_share_obs)   
        self.obs = np.array(self.obs).astype(np.float32)
        self.share_obs = np.array(self.share_obs).astype(np.float32) 
        self.recurrent_hidden_states = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        self.recurrent_hidden_states_critic = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        print(self.obs)
        print(self.share_obs)
        while self.duration is None or time.time() < self.end_time:
            values = []
            actions= []
            recurrent_hidden_statess = []
            recurrent_hidden_statess_critic = []
            with torch.no_grad():                
                for agent_id in range(self.num_agents):
                    self.policies[0].eval()
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = self.policies[0].act(agent_id,
                    torch.tensor(self.share_obs[:,agent_id,:]), 
                    torch.tensor(self.obs[:,agent_id,:]), 
                    torch.tensor(self.recurrent_hidden_states[:,agent_id,:]), 
                    torch.tensor(self.recurrent_hidden_states_critic[:,agent_id,:]),
                    torch.tensor(self.masks[:,agent_id,:]))
                    values.append(value.detach().cpu().numpy())
                    actions.append(action.detach().cpu().numpy())
                    recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
                    recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())

            action_movement = []
            action_pull = []
            action_glueall = []
            for agent_id in range(self.num_agents):
                action_movement.append(actions[agent_id][0][:self.action_movement_dim[agent_id]])
                action_glueall.append(int(actions[agent_id][0][self.action_movement_dim[agent_id]]))
                if 'action_pull' in self.env.action_space.spaces.keys():
                    action_pull.append(int(actions[agent_id][0][-1]))
            action_movement = np.stack(action_movement, axis = 0)
            action_glueall = np.stack(action_glueall, axis = 0)
            if 'action_pull' in self.env.action_space.spaces.keys():
                action_pull = np.stack(action_pull, axis = 0)                             
            one_env_action = {'action_movement': action_movement, 'action_pull': action_pull, 'action_glueall': action_glueall}
            print(action_pull)
            self.dict_obs, rew, done, env_info = self.env.step(one_env_action)
            print(self.dict_obs)
            self.total_rew += rew
            self.obs = []
            self.share_obs = []   
            for i, key in enumerate(self.order_obs):
                if key in self.env.observation_space.spaces.keys():             
                    if self.mask_order_obs[i] == None:
                        temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                        temp_obs = temp_share_obs.copy()
                    else:
                        temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                        temp_mask = self.dict_obs[self.mask_order_obs[i]].copy()
                        temp_obs = self.dict_obs[key].copy()
                        mins_temp_mask = ~temp_mask
                        temp_obs[mins_temp_mask]=np.zeros(((mins_temp_mask).sum(),temp_obs.shape[2]))                       
                        temp_obs = temp_obs.reshape(self.num_agents,-1) 
                    if i == 0:
                        reshape_obs = temp_obs.copy()
                        reshape_share_obs = temp_share_obs.copy()
                    else:
                        reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                        reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
            self.obs.append(reshape_obs)
            self.share_obs.append(reshape_share_obs)   
            self.obs = np.array(self.obs).astype(np.float32)
            self.share_obs = np.array(self.share_obs).astype(np.float32)
            self.recurrent_hidden_states = np.array(recurrent_hidden_statess).transpose(1,0,2)
            self.recurrent_hidden_states_critic = np.array(recurrent_hidden_statess_critic).transpose(1,0,2)
            if done or env_info.get('discard_episode', False):
                self.reset_increment()

            if self.display_window:
                self.add_overlay(const.GRID_TOPRIGHT, "Reset env; (current seed: {})".format(self.seed), "N - next / P - previous ")
                self.add_overlay(const.GRID_TOPRIGHT, "Reward", str(self.total_rew))
                if hasattr(self.env.unwrapped, "viewer_stats"):
                    for k, v in self.env.unwrapped.viewer_stats.items():
                        self.add_overlay(const.GRID_TOPRIGHT, k, str(v))

                self.env.render()

    def reset_increment(self):
        self.total_rew_avg = (self.n_episodes * self.total_rew_avg + self.total_rew) / (self.n_episodes + 1)
        self.n_episodes += 1
        print(f"Reward: {self.total_rew} (rolling average: {self.total_rew_avg})")
        self.total_rew = 0.0
        self.seed += 1
        self.env.seed(self.seed)
        self.dict_obs = self.env.reset()
        self.obs = []
        self.share_obs = []   
        for i, key in enumerate(self.order_obs):
            if key in self.env.observation_space.spaces.keys():             
                if self.mask_order_obs[i] == None:
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_obs = temp_share_obs.copy()
                else:
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_mask = self.dict_obs[self.mask_order_obs[i]].copy()
                    temp_obs = self.dict_obs[key].copy()
                    mins_temp_mask = ~temp_mask
                    temp_obs[mins_temp_mask]=np.zeros(((mins_temp_mask).sum(),temp_obs.shape[2]))                       
                    temp_obs = temp_obs.reshape(self.num_agents,-1) 
                if i == 0:
                    reshape_obs = temp_obs.copy()
                    reshape_share_obs = temp_share_obs.copy()
                else:
                    reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                    reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
        self.obs.append(reshape_obs)
        self.share_obs.append(reshape_share_obs)   
        self.obs = np.array(self.obs).astype(np.float32)
        self.share_obs = np.array(self.share_obs).astype(np.float32) 
        self.recurrent_hidden_states = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        self.recurrent_hidden_states_critic = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        #for policy in self.policies:
        #    policy.reset()
        if hasattr(self.env, "reset_goal"):
            self.goal = self.env.reset_goal()
        self.update_sim(self.env.unwrapped.sim)

class PolicyViewer_bc(MjViewer):
    '''
    PolicyViewer runs a policy with an environment and optionally displays it.
        env - environment to run policy in
        policy - policy object to run
        display_window - if true, show the graphical viewer
        seed - environment seed to view
        duration - time in seconds to run the policy, run forever if duration=None
    '''
    @store_args
    def __init__(self, env, policies, display_window=True, seed=None, duration=None):
        if seed is None:
            self.seed = env.seed()[0]
        else:
            self.seed = seed
            env.seed(seed)
        self.total_rew = 0.0
        self.dict_obs = env.reset()
        #for policy in self.policies:
        #    policy.reset()
        #assert env.metadata['n_actors'] % len(policies) == 0
        if hasattr(env, "reset_goal"):
            self.goal = env.reset_goal()
        super().__init__(self.env.unwrapped.sim)
        # TO DO: remove circular dependency on viewer object. It looks fishy.
        self.env.unwrapped.viewer = self
        if self.render and self.display_window:
            self.env.render()

    def key_callback(self, window, key, scancode, action, mods):
        super().key_callback(window, key, scancode, action, mods)
        # Trigger on keyup only:
        if action != glfw.RELEASE:
            return
        # Increment experiment seed
        if key == glfw.KEY_N:
            self.reset_increment()
        # Decrement experiment trial
        elif key == glfw.KEY_P:
            print("Pressed P")
            self.seed = max(self.seed - 1, 0)
            self.env.seed(self.seed)
            self.ob = self.env.reset()
            for policy in self.policies:
                policy.reset()
            if hasattr(self.env, "reset_goal"):
                self.goal = self.env.reset_goal()
            self.update_sim(self.env.unwrapped.sim)

    def run(self):
        self.action_movement_dim = []
        '''
        self.order_obs = ['agent_qpos_qvel','box_obs','ramp_obs','food_obs','observation_self']    
        self.mask_order_obs = ['mask_aa_obs','mask_ab_obs','mask_ar_obs','mask_af_obs',None]
        '''
        '''
        self.order_obs = ['box_obs','ramp_obs','construction_site_obs','vector_door_obs', 'observation_self']    
        self.mask_order_obs = [None,'mask_ar_obs',None,None,None]
        '''
        self.order_obs = ['agent_qpos_qvel','box_obs','ramp_obs','construction_site_obs','vector_door_obs', 'observation_self']    
        self.mask_order_obs = [None,None,'mask_ar_obs',None,None,None]

        self.num_agents = 2
        for agent_id in range(self.num_agents):
            action_movement = self.env.action_space['action_movement'][agent_id].nvec
            self.action_movement_dim.append(len(action_movement))
        self.masks = np.ones((1, self.num_agents, 1)).astype(np.float32)
        if self.duration is not None:
            self.end_time = time.time() + self.duration
        self.total_rew_avg = 0.0
        self.n_episodes = 0
        self.obs = []
        self.share_obs = []   
        for i, key in enumerate(self.order_obs):
            if key in self.env.observation_space.spaces.keys():             
                if self.mask_order_obs[i] == None:          
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_obs = temp_share_obs.copy()
                else:
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_mask = self.dict_obs[self.mask_order_obs[i]].copy()
                    temp_obs = self.dict_obs[key].copy()
                    mins_temp_mask = ~temp_mask
                    temp_obs[mins_temp_mask]=np.zeros(((mins_temp_mask).sum(),temp_obs.shape[2]))                       
                    temp_obs = temp_obs.reshape(self.num_agents,-1) 
                if i == 0:
                    reshape_obs = temp_obs.copy()
                    reshape_share_obs = temp_share_obs.copy()
                else:
                    reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                    reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                   
        self.obs.append(reshape_obs)
        self.share_obs.append(reshape_share_obs)   
        self.obs = np.array(self.obs).astype(np.float32)
        self.share_obs = np.array(self.share_obs).astype(np.float32) 
        self.recurrent_hidden_states = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        self.recurrent_hidden_states_critic = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        while self.duration is None or time.time() < self.end_time:
            values = []
            actions= []
            recurrent_hidden_statess = []
            recurrent_hidden_statess_critic = []
            with torch.no_grad():                
                for agent_id in range(self.num_agents):
                    self.policies[0].eval()
                    value, action, action_log_prob, recurrent_hidden_states, recurrent_hidden_states_critic = self.policies[0].act(agent_id,
                    torch.tensor(self.share_obs[:,agent_id,:]), 
                    torch.tensor(self.obs[:,agent_id,:]), 
                    torch.tensor(self.recurrent_hidden_states[:,agent_id,:]), 
                    torch.tensor(self.recurrent_hidden_states_critic[:,agent_id,:]),
                    torch.tensor(self.masks[:,agent_id,:]))
                    values.append(value.detach().cpu().numpy())
                    actions.append(action.detach().cpu().numpy())
                    recurrent_hidden_statess.append(recurrent_hidden_states.detach().cpu().numpy())
                    recurrent_hidden_statess_critic.append(recurrent_hidden_states_critic.detach().cpu().numpy())

            action_movement = []
            action_pull = []
            action_glueall = []
            for agent_id in range(self.num_agents):
                action_movement.append(actions[agent_id][0][:self.action_movement_dim[agent_id]])
                action_glueall.append(int(actions[agent_id][0][self.action_movement_dim[agent_id]]))
                if 'action_pull' in self.env.action_space.spaces.keys():
                    action_pull.append(int(actions[agent_id][0][-1]))
            action_movement = np.stack(action_movement, axis = 0)
            action_glueall = np.stack(action_glueall, axis = 0)
            if 'action_pull' in self.env.action_space.spaces.keys():
                action_pull = np.stack(action_pull, axis = 0)                             
            one_env_action = {'action_movement': action_movement, 'action_pull': action_pull, 'action_glueall': action_glueall}
            self.dict_obs, rew, done, env_info = self.env.step(one_env_action)
            #print(self.dict_obs)
            self.total_rew += rew
            self.obs = []
            self.share_obs = []   
            for i, key in enumerate(self.order_obs):
                if key in self.env.observation_space.spaces.keys():             
                    if self.mask_order_obs[i] == None:
                        temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                        temp_obs = temp_share_obs.copy()
                    else:
                        temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                        temp_mask = self.dict_obs[self.mask_order_obs[i]].copy()
                        temp_obs = self.dict_obs[key].copy()
                        mins_temp_mask = ~temp_mask
                        temp_obs[mins_temp_mask]=np.zeros(((mins_temp_mask).sum(),temp_obs.shape[2]))                       
                        temp_obs = temp_obs.reshape(self.num_agents,-1) 
                    if i == 0:
                        reshape_obs = temp_obs.copy()
                        reshape_share_obs = temp_share_obs.copy()
                    else:
                        reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                        reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
            self.obs.append(reshape_obs)
            self.share_obs.append(reshape_share_obs)   
            self.obs = np.array(self.obs).astype(np.float32)
            self.share_obs = np.array(self.share_obs).astype(np.float32)
            self.recurrent_hidden_states = np.array(recurrent_hidden_statess).transpose(1,0,2)
            self.recurrent_hidden_states_critic = np.array(recurrent_hidden_statess_critic).transpose(1,0,2)
            if done or env_info.get('discard_episode', False):
                self.reset_increment()

            if self.display_window:
                self.add_overlay(const.GRID_TOPRIGHT, "Reset env; (current seed: {})".format(self.seed), "N - next / P - previous ")
                self.add_overlay(const.GRID_TOPRIGHT, "Reward", str(self.total_rew))
                if hasattr(self.env.unwrapped, "viewer_stats"):
                    for k, v in self.env.unwrapped.viewer_stats.items():
                        self.add_overlay(const.GRID_TOPRIGHT, k, str(v))

                self.env.render()

    def reset_increment(self):
        self.total_rew_avg = (self.n_episodes * self.total_rew_avg + self.total_rew) / (self.n_episodes + 1)
        self.n_episodes += 1
        print(f"Reward: {self.total_rew} (rolling average: {self.total_rew_avg})")
        self.total_rew = 0.0
        self.seed += 1
        self.env.seed(self.seed)
        self.dict_obs = self.env.reset()
        self.obs = []
        self.share_obs = []   
        for i, key in enumerate(self.order_obs):
            if key in self.env.observation_space.spaces.keys():             
                if self.mask_order_obs[i] == None:
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_obs = temp_share_obs.copy()
                else:
                    temp_share_obs = self.dict_obs[key].reshape(self.num_agents,-1).copy()
                    temp_mask = self.dict_obs[self.mask_order_obs[i]].copy()
                    temp_obs = self.dict_obs[key].copy()
                    mins_temp_mask = ~temp_mask
                    temp_obs[mins_temp_mask]=np.zeros(((mins_temp_mask).sum(),temp_obs.shape[2]))                       
                    temp_obs = temp_obs.reshape(self.num_agents,-1) 
                if i == 0:
                    reshape_obs = temp_obs.copy()
                    reshape_share_obs = temp_share_obs.copy()
                else:
                    reshape_obs = np.concatenate((reshape_obs,temp_obs),axis=1) 
                    reshape_share_obs = np.concatenate((reshape_share_obs,temp_share_obs),axis=1)                    
        self.obs.append(reshape_obs)
        self.share_obs.append(reshape_share_obs)   
        self.obs = np.array(self.obs).astype(np.float32)
        self.share_obs = np.array(self.share_obs).astype(np.float32) 
        self.recurrent_hidden_states = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        self.recurrent_hidden_states_critic = np.zeros((1, self.num_agents, 64)).astype(np.float32)
        #for policy in self.policies:
        #    policy.reset()
        if hasattr(self.env, "reset_goal"):
            self.goal = self.env.reset_goal()
        self.update_sim(self.env.unwrapped.sim)