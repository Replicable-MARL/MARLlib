# Meta-Drive in Ray

This is Meta-Drive baseline with **ray[rllib]**

- Remove unnecessary features/functions, only keep core function, 
- Easier to read and extend compared to original version in https://github.com/decisionforce/CoPO

## Getting Started
Install Ray
> pip install ray==1.8.0 # version is important

Install Meta-Drive environment
> pip install metadrive-simulator==0.2.3

Install dependencies including(version is not strict)

        "yapf==0.30.0",
        "tensorboardX",
        "gym==0.19.0"

## current support algo
- PG
- A2C
- MAA2C
- DDPG
- MADDPG
- PPO
- MAPPO
  
### with neural arch
- GRU
- LSTM



