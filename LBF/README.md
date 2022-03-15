# LBF in Ray

This is lb-foraging(LBF) baseline built with **ray[rllib]**

## Getting Started

Install LBF
> pip install lbforaging==1.0.15

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
- VDA2C-SUM/MIX
- VDPPO-SUM/MIX

Note: VDN/QMIX/VDA2C/VDPPO only work with setting *coop-force = True* in **config_lbf.py** or script command
  
### with neural arch
- GRU
- LSTM



