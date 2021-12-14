# Scaling up MARL in Ray

This is **SMAC** environment built for **ray[rllib]**
> pip install ray=1.8.0 # version is important

Please annotate one line source code to avoid parallel env seed bug
> ray.rllib.evaluation.rollout_worker.py line 508 


### current support algo
- R2D2(IQL)
- VDN
- QMIX
- PG
- A2C
- A3C
- MAA2C
- PPO
- MAPPO
  
### with neural arch
- GRU
- LSTM
- UPDeT

# TODO
### incorporate more env
- Neural-mmo (have instance in Ray)
- Meta-Drive (have instance in Ray)
- LBF
- RWARE
- MPE
- MALMO
- Pommerman
- DM Lab
- Derk's Gym
- Hanabi

