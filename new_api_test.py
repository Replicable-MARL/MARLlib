import marl

env = marl.make_env(environment_name='smac', map_name='3m', mask_flag=True, global_state_flag=True,
                    opp_action_in_cc=True, fixed_batch_timesteps=3200, difficulty="7", reward_scale_rate=20)

env.set_ray(local_model=True, stop_iters=100000)

happo = marl.algos.HAPPO(use_gae=True, gamma=0.99, _lambda=1.0, batch_episode=10, kl_coeff=0.2, num_sgd_iter=5, grad_clip=10, clip_param=0.3, critic_lr=0.0005,
                         vf_loss_coeff=1.0, lr=0.0000005, entropy_coeff=0.01, vf_clip_param=10.0,
                         batch_mode="complete_episodes")

happo.fit_online(env, stop={'episode_reward_mean': 2000}, num_workers=4)
