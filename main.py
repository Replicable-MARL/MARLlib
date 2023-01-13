import marl

'''
support commandline parameter insert to running
# some instances
1 ray/rllib config: --ray_args.local_mode
2 environments: --env_args.difficulty=6 
3 algorithms: --algo_args.num_sgd_iter=6
python main.py --ray_args.local_mode --env_args.difficulty=6  --algo_args.num_sgd_iter=6
'''

# prepare the environment
env = marl.make_env(environment_name="mamujoco", map_name="2AgentAnt")
# can add extra env params. remember to check env configuration before use
# env = marl.make_env(environment_name='smac', map_name='3m', difficulty="6", reward_scale_rate=15)

# initialize algorithm and load hyperparameters
mappo = marl.algos.mappo(hyperparam_source='common')
# can add extra algorithm params. remember to check algo hyperparams before use
# mappo = marl.algos.MAPPO(hyperparam_source='common', use_gae=True,  batch_episode=10, kl_coeff=0.2, num_sgd_iter=3)

# start learning + extra experiment settings if needed. remember to check ray.yaml before use
mappo.fit(env, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000000}, local_mode=True, num_gpus=1,
          num_workers=2, share_policy='all', checkpoint_freq=5)
