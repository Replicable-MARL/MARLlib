# RWARE in Ray

This is Multi-Robot Warehouse(RWARE) baseline built with **ray[rllib]**

## Getting Started

Install RWARE
> pip install rware==1.0.1

Install Ray
> pip install ray==1.8.0 # version is important

Please annotate one line source code to avoid parallel env seed bug (some env need this)
at ray.rllib.evaluation.rollout_worker.py line 508

> _update_env_seed_if_necessary(self.env, seed, worker_index, 0)

## current support algo
- R2D2(IQL)
- VDN
- QMIX
- PG
- A2C
- A3C
- MAA2C
- PPO
- MAPPO
- COMA

  
### with neural arch
- GRU
- LSTM



