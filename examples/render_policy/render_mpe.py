from marllib import marl

# prepare the environment
#env = marl.make_env(environment_name="hanabi", map_name="Hanabi-Very-Small")
env = marl.make_env(environment_name="mpe", map_name="simple_spread")
# can add extra env params. remember to check env configuration before use
# env = marl.make_env(environment_name='smac', map_name='3m', difficulty="6", reward_scale_rate=15)

# initialize algorithm and load hyperparameters
mappo = marl.algos.mappo(hyperparam_source="test")
# can add extra algorithm params. remember to check algo_config hyperparams before use
# mappo = marl.algos.MAPPO(hyperparam_source='common', use_gae=True,  batch_episode=10, kl_coeff=0.2, num_sgd_iter=3)

# build agent model based on env + algorithms + user preference if checked available
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# rendering after 1 training iteration
mappo.render(env, model, local_mode=True, num_gpus=0, num_workers=0, share_policy='all',
              restore_path='checkpoint_003050/checkpoint-3050', checkpoint_end=False)
