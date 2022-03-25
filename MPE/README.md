# MPE in Ray

This is MPE baseline built with **ray[rllib]**

## Getting Started
Install Ray
> pip install ray==1.8.0 # version is important

Please annotate one line source code to avoid parallel env seed bug (some env need this)
at ray.rllib.evaluation.rollout_worker.py line 508

> _update_env_seed_if_necessary(self.env, seed, worker_index, 0)

## current support algo
- R2D2(IQL) [D]
- PG [D+C]
- A2C [D+C]
- A3C [D+C]
- MAA2C [D+C]
- DDPG [C]
- MADDPG [C]
- PPO [D+C]
- MAPPO [D+C]
- COMA [D]
- VDN [D]
- QMIX [D]  
- VDA2C-SUM/MIX [D+C]
- VDPPO-SUM/MIX [D+C]

Note here D for Discrete, C for Continues
  
### with neural arch
- GRU
- LSTM
- MLP(DDPG, MADDPG)



