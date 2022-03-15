# Hanabi in Ray

This is Hanabi baseline built with **ray[rllib]**

## Getting Started

Install Hanabi
> pip install hanabi_learning_environment

If bug reported related to *enum* package
> pip uninstall -y enum34

Install Ray
> pip install ray==1.8.0 # version is important

Please annotate one line source code to avoid parallel env seed bug (some env need this)
at ray.rllib.evaluation.rollout_worker.py line 508

> _update_env_seed_if_necessary(self.env, seed, worker_index, 0)

## current support algo
- R2D2(IQL)
- PG
- A2C
- A3C
- MAA2C
- PPO
- MAPPO
- COMA

Note: Value decomposition methods are not suitable for turn-based game
  
### with neural arch
- GRU
- LSTM



