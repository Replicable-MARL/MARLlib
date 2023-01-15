import marl

# prepare the environment & initialize algorithm
env = marl.make_env(environment_name="mamujoco", map_name="2AgentAnt")
vda2c = marl.algos.vda2c(hyperparam_source='common')

# rendering after 1 training iteration
vda2c.render(env, local_mode=True, num_gpus=1, num_workers=2, share_policy='all',
              restore_path='vda2c_2AgentAnt_checkpoint_000015/checkpoint-15', checkpoint_end=False)
