# Lighter Neural-MMO in Ray

This is Neural-MMO baseline with **ray[rllib]**

- Remove unnecessary features/functions, only keep core function, 
- Easier to read and extend compared to original version in https://gitlab.aicrowd.com/neural-mmo/neural-mmo-starter-kit

> pip install neural-mmo[rllib]==1.5.2.2 # version is important # this will force install Ray==1.4.0

> pip uninstall ray # remove old version 1.4.0

> pip install ray==1.8.0 # version is important

Please add one line source code in **copy.py** according to
https://bugs.python.org/issue38293 to avoid copy bug of @property

Please annotate one line source code to avoid parallel env seed bug (some env need this)
at ray.rllib.evaluation.rollout_worker.py line 508

> _update_env_seed_if_necessary(self.env, seed, worker_index, 0)


### current support algo
- PG
- A2C
- A3C
- PPO
  
### with neural arch
- GRU
- LSTM



