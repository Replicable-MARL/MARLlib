import marl

# prepare the environment
env = marl.make_env(environment_name="mamujoco", map_name="2AgentAnt")

# initialize algorithm and load hyperparameters
mappo = marl.algos.mappo(hyperparam_source='common')

# start learning + extra experiment settings if needed
mappo.render(env, stop={'training_iteration': 1}, local_mode=True, num_gpus=1, num_workers=2, share_policy='all',
          restore_path='checkpoint_000015/checkpoint-15', checkpoint_end=False)
