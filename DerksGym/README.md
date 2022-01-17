# Derk's Gym in Ray

This is [Derk's Gym](https://gym.derkgame.comnump) baseline built with **ray[rllib]**

## Getting Started

Install LBF
> pip install gym-derk==1.1.1

Install Ray
> pip install ray==1.8.0 # version is important

Please annotate one line source code to avoid parallel env seed bug (some env need this)
at ray.rllib.evaluation.rollout_worker.py line 508

> _update_env_seed_if_necessary(self.env, seed, worker_index, 0)

## current support algo
- PG
- A2C
- MAA2C
- PPO
- MAPPO
 
### with neural arch
- GRU
- LSTM

### customized your environment

please refer to:
- http://docs.gym.derkgame.com/#creature-config
- http://docs.gym.derkgame.com/#items
- modify line 11-14 in derk_rllib.py

      self.env = DerkEnv(
        n_arenas=1,
        turbo_mode=True,
        agent_server_args={"port": random.randint(8888, 9999)})




