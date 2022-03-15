# Pommerman in Ray

This is Pommerman(playground) baseline built with **ray[rllib]**

## Getting Started

Install Pommerman
> git clone https://github.com/MultiAgentLearning/playground

> cd playground

> pip install .


Install Ray
> pip install ray==1.8.0 # version is important

Please annotate one line source code to avoid parallel env seed bug (some env need this)
at ray.rllib.evaluation.rollout_worker.py line 508

> _update_env_seed_if_necessary(self.env, seed, worker_index, 0)

Pommerman require gym=0.10.11, which is a very old version that RLlib not use anymore.

Here we solve this conflict by modifying some source code of Pommerman as follows:
(you can find the replace file in *source* directory) 

Pattern: source package file -> replace file

- **pommerman/graphics.py**  ->  **graphics.py**
- **pommerman/\_\_init\_\_.py**  ->  **\_\_init\_\_.py** 
- **pommerman/forward_model.py**  ->  **forward_model.py** 
- **pommerman/env/v0.py**  ->  **v0.py** 

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

Note: 
- VDN and QMIX only work with map **PommeTeamCompetition-v0**
- COMA, MAPPO and MAA2C can not work with existing of rule-based agent (require all the agents to be trainable)
- Additional restrictions on trainable/human-rule/random agents' specification on different maps. You will learn from the error log printed. 
  
### with neural arch
- CNN
- CNN+GRU
- CNN+LSTM

Note:
- only CNN_GRU is supported for QMIX and VDN





