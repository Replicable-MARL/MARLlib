Currently, 10 environments are available for Independent Learning

- Football
- MPE 
- SMAC
- mamujoco
- RWARE
- LBF 
- Pommerman
- Magent
- MetaDrive
- Hanabi

```
CUDA_VISIBLE_DEVICES=1 python marl/run.py --algo_config=a2c --env-config=smac with env_args.map_name=3m
```

Currently, 7 environments are available for Value Decomposition

- Football
- MPE 
- SMAC
- mamujoco
- RWARE
- LBF 
- Pommerman

```
CUDA_VISIBLE_DEVICES=1 python marl/run.py --algo_config=qmix --env-config=smac with env_args.map_name=3m
```


Currently, 9 environments are available for Centralized Critic

- Football
- MPE 
- SMAC
- mamujoco
- RWARE
- LBF 
- Pommerman
- Magent
- Hanabi

```
CUDA_VISIBLE_DEVICES=1 python marl/run.py --algo_config=mappo --env-config=smac with env_args.map_name=3m
```

