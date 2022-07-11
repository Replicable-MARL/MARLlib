

<div align="center">
<img src=image/logo1.png width=70% />
</div>

<h1 align="center"> MARLlib: The MARL Extension for RLlib </h1>

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Replicable-MARL/MARLlib/blob/main/LICENSE)

**Multi-Agent RLlib (MARLlib)** is a *Multi-Agent Reinforcement Learning benchmark* based on [**Ray**](https://github.com/ray-project/ray) and one of its toolkits [**RLlib**](https://github.com/ray-project/ray/tree/master/rllib). 
It provides MARL research community a unified platform for developing and evaluating the new ideas in various multi-agent environments.
There are four core features of **MARLlib**. 

- it collects most of the existing MARL algorithms widely acknowledged by the community and unifies them under one framework. 
- it gives a solution that enables different multi-agent environments using the same interface to interact with the agents.
- it guarantees excellent efficiency in both the training and sampling process.
- it provides trained results, including learning curves and pretrained models specific to each task and algorithm's combination, with finetuned hyper-parameters to guarantee credibility. 

<div align="center">
<img src=image/overview.png width=100% />
</div>

**Project Website**: https://sites.google.com/view/marllib


[comment]: <> (--------------------------------------------------------------------------------)

[comment]: <> (The README is organized as follows:)

[comment]: <> (- [**I. Overview**]&#40;#Part-I.-Overview&#41;)

[comment]: <> (- [**II. Environment**]&#40;#Part-II.-Environment&#41;)

[comment]: <> (- [**III. Algorithm**]&#40;#Part-III.-Algorithm&#41;)

[comment]: <> (- [**IV. Getting started & Examples**]&#40;#Part-VI.-Getting-started&#41;)

[comment]: <> (- [**V. Contribute New Environment**]&#40;&#41;)

[comment]: <> (- [**VI. Bug Shooting**]&#40;#Part-VI.-Bug-Shooting&#41;)

## Overview

We collected most of the existing multi-agent environment and multi-agent reinforcement learning algorithms and unified them under one framework based on [**Ray**](https://github.com/ray-project/ray) 's [**RLlib**](https://github.com/ray-project/ray/tree/master/rllib) to boost the MARL research. 

The MARL baselines include **independence learning (IQL, A2C, DDPG, TRPO, PPO)**, **centralized critic learning (COMA, MADDPG, MAPPO, HATRPO)**, and **value decomposition (QMIX, VDN, FACMAC, VDA2C)** are all implemented.

Popular environments like **SMAC, MaMujoco, and Google Research Football** are provided with a unified interface.

The algorithm code and environment code are fully separated. Changing the environment needs no modification on the algorithm side and vice versa.

Here we provide a table for comparison of MARLlib and before benchmarks.

|   Benchmark   | Github Stars | Learning Mode | Available Env | Algorithm Type | Algorithm Number | Continues Control | Asynchronous Interact | Distributed Training |          Framework          | Last Update |
|:-------------:|:-------------:|:-------------:|:-------------:|:--------------:|:----------------:|:-----------------:|:---------------------:|:--------------------:|:---------------------------:|:---------------------------:
|     [PyMARL](https://github.com/oxwhirl/pymarl) | [![GitHub stars](https://img.shields.io/github/stars/oxwhirl/pymarl)](https://github.com/oxwhirl/pymarl/stargazers)    |       CP      |       1       |       VD       |         5        |                   |                       |                      |              *              | ![GitHub last commit](https://img.shields.io/github/last-commit/oxwhirl/pymarl?label=last%20update) |
|    [PyMARL2](https://github.com/hijkzzz/pymarl2) | [![GitHub stars](https://img.shields.io/github/stars/hijkzzz/pymarl2)](https://github.com/hijkzzz/pymarl2/stargazers)    |       CP      |       1       |       VD       |         12        |                   |                       |                      | [PyMARL](https://github.com/oxwhirl/pymarl) | ![GitHub last commit](https://img.shields.io/github/last-commit/hijkzzz/pymarl2?label=last%20update) |
|   [MARL-Algorithms](https://github.com/starry-sky6688/MARL-Algorithms)| [![GitHub stars](https://img.shields.io/github/stars/starry-sky6688/MARL-Algorithms)](https://github.com/starry-sky6688/MARL-Algorithms/stargazers)  |       CP      |       1       |     VD+Comm    |         9        |                   |                       |                      |              *              | ![GitHub last commit](https://img.shields.io/github/last-commit/starry-sky6688/MARL-Algorithms?label=last%20update) |
|    [EPyMARL](https://github.com/uoe-agents/epymarl)| [![GitHub stars](https://img.shields.io/github/stars/uoe-agents/epymarl)](https://github.com/hijkzzz/uoe-agents/epymarl/stargazers)    |       CP      |       4       |    IL+VD+CC    |        10        |                   |                       |                      |            [PyMARL](https://github.com/oxwhirl/pymarl)           | ![GitHub last commit](https://img.shields.io/github/last-commit/uoe-agents/epymarl?label=last%20update) |
| [Marlbenchmark](https://github.com/marlbenchmark/on-policy)| [![GitHub stars](https://img.shields.io/github/stars/marlbenchmark/on-policy)](https://github.com/marlbenchmark/on-policy/stargazers) |     CP+CL     |       4       |      VD+CC     |         5        |         :heavy_check_mark:         |                       |                      | [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) | ![GitHub last commit](https://img.shields.io/github/last-commit/marlbenchmark/on-policy?label=last%20update) |
| [MAlib](https://github.com/sjtu-marl/malib) | [![GitHub stars](https://img.shields.io/github/stars/sjtu-marl/malib)](https://github.com/hijkzzz/sjtu-marl/malib/stargazers) | SP | 8 | SP | 9 | :heavy_check_mark: |  |  | * | ![GitHub last commit](https://img.shields.io/github/last-commit/sjtu-marl/malib?label=last%20update)
|    [MARLlib](https://github.com/Replicable-MARL/MARLlib)|[![GitHub stars](https://img.shields.io/github/stars/Replicable-MARL/MARLlib)](https://github.com/Replicable-MARL/MARLlib/stargazers) |  CP+CL+CM+MI  |       10      |    IL+VD+CC    |        18        |         :heavy_check_mark:         |           :heavy_check_mark:           |           :heavy_check_mark:         |            [Ray/RLlib](https://docs.ray.io/en/releases-1.8.0/)            | ![GitHub last commit](https://img.shields.io/github/last-commit/Replicable-MARL/MARLlib?label=last%20update) |

CP, CL, CM, and MI represent cooperative, collaborative, competitive, and mixed task learning modes. 
IL, VD, and CC represent independent learning, value decomposition, and centralized critic categorization. SP represents self-play. 
Comm represents communication-based learning. 
Asterisk denotes that the benchmark uses its framework.



The tutorial of RLlib can be found at this [link](https://docs.ray.io/en/releases-1.8.0/).
Fast examples can be found at this [link](https://docs.ray.io/en/releases-1.8.0/rllib-examples.html). 
These will help you quickly dive into RLlib.

We hope MARLlib can benefit everyone interested in RL/MARL.



## Environment

#### Supported Multi-agent Environments / Tasks

Most of the popular environment in MARL research has been incorporated in this benchmark:

| Env Name | Learning Mode | Observability | Action Space | Observations |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| [LBF](https://github.com/semitable/lb-foraging)  | Mixed | Both | Discrete | Discrete  |
| [RWARE](https://github.com/semitable/robotic-warehouse)  | Collaborative | Partial | Discrete | Discrete  |
| [MPE](https://github.com/openai/multiagent-particle-envs)  | Mixed | Both | Both | Continuous  |
| [SMAC](https://github.com/oxwhirl/smac)  | Cooperative | Partial | Discrete | Continuous |
| [MetaDrive](https://github.com/decisionforce/metadrive)  | Collaborative | Partial | Continuous | Continuous |
|[MAgent](https://www.pettingzoo.ml/magent) | Mixed | Partial | Discrete | Discrete |
| [Pommerman](https://github.com/MultiAgentLearning/playground)  | Mixed | Both | Discrete | Discrete |
| [MaMujoco](https://github.com/schroederdewitt/multiagent_mujoco)  | Cooperative | Partial | Continuous | Continuous |
| [GRF](https://github.com/google-research/football)  | Collaborative | Full | Discrete | Continuous |
| [Hanabi](https://github.com/deepmind/hanabi-learning-environment) | Cooperative | Partial | Discrete | Discrete |

Each environment has a readme file, standing as the instruction for this task, talking about env settings, installation, and some important notes.

## Algorithm

We provide three types of MARL algorithms as our baselines including:

**Independent Learning:** 
IQL
DDPG
PG
A2C
TRPO
PPO

**Centralized Critic:**
COMA 
MADDPG 
MAAC 
MAPPO
MATRPO
HATRPO
HAPPO

**Value Decomposition:**
VDN
QMIX
FACMAC
VDAC
VDPPO

Here is a chart describing the characteristics of each algorithm:

| Algorithm | Support Task Mode | Need Global State | Action | Learning Mode  | Type |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| IQL*  | Mixed | No | Discrete | Independent Learning | Off Policy
| [PG](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)  | Mixed | No | Both | Independent Learning | On Policy
| [A2C](https://arxiv.org/abs/1602.01783)  | Mixed | No | Both | Independent Learning | On Policy
| [DDPG](https://arxiv.org/abs/1509.02971)  | Mixed | No | Continuous | Independent Learning | Off Policy
| [TRPO](http://proceedings.mlr.press/v37/schulman15.pdf)  | Mixed | No | Both | Independent Learning | On Policy
| [PPO](https://arxiv.org/abs/1707.06347)  | Mixed | No | Both | Independent Learning | On Policy
| [COMA](https://ojs.aaai.org/index.php/AAAI/article/download/11794/11653)  | Mixed | Yes | Both | Centralized Critic | On Policy
| [MADDPG](https://arxiv.org/abs/1706.02275)  | Mixed | Yes | Continuous | Centralized Critic | Off Policy
| MAA2C*  | Mixed | Yes | Both | Centralized Critic | On Policy
| MATRPO*  | Mixed | Yes | Both | Centralized Critic | On Policy
| [MAPPO](https://arxiv.org/abs/2103.01955)  | Mixed | Yes | Both | Centralized Critic | On Policy
| [HATRPO](https://arxiv.org/abs/2109.11251)  | Cooperative | Yes | Both | Centralized Critic | On Policy
| [HAPPO](https://arxiv.org/abs/2109.11251)  | Cooperative | Yes | Both | Centralized Critic | On Policy
| [VDN](https://arxiv.org/abs/1706.05296) | Cooperative | No | Discrete | Value Decomposition | Off Policy
| [QMIX](https://arxiv.org/abs/1803.11485)  | Cooperative | Yes | Discrete | Value Decomposition | Off Policy
| [FACMAC](https://arxiv.org/abs/2003.06709)  | Cooperative | Yes | Continuous | Value Decomposition | Off Policy
| [VDAC](https://arxiv.org/abs/2007.12306)  | Cooperative | Yes | Both | Value Decomposition | On Policy
| VDPPO*| Cooperative | Yes | Both | Value Decomposition | On Policy

*IQL* is the multi-agent version of Q learning.
*MAA2C* and *MATRPO* are the centralized version of A2C and TRPO.
*VDPPO* is the value decomposition version of PPO.

**Current Task & Available algorithm mapping**: Y for available, N for not suitable, P for partially available on some scenarios.
(Note: in our code, independent algorithms may not have **I** as prefix. For instance, PPO = IPPO)

| Env w Algorithm | IQL | PG | A2C | DDPG | TRPO | PPO | COMA | MADDPG | MAAC | MATRPO | MAPPO | HATRPO | HAPPO| VDN | QMIX | FACMAC| VDAC | VDPPO 
| ---- | ---- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |  ---- | ---- | ---- | ---- | ---- |
| LBF         | Y | Y | Y | N | Y | Y | Y | N | Y | Y | Y | Y | Y | P | P | P | P | P |
| RWARE       | Y | Y | Y | N | Y | Y | Y | N | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| MPE         | P | Y | Y | P | Y | Y | P | P | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| SMAC        | Y | Y | Y | N | Y | Y | Y | N | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| MetaDrive  | N | Y | Y | Y | Y | Y | N | N | N | N | N | N | N | N | N | N | N | N |
| MAgent      | Y | Y | Y | N | Y | Y | Y | N | Y | Y | Y | Y | Y | N | N | N | N | N |
| Pommerman   | Y | Y | Y | N | Y | Y | P | N | Y | Y | Y | Y | Y | P | P | P | P | P |
| MaMujoco    | N | Y | Y | Y | Y | Y | N | Y | Y | Y | Y | Y | Y | N | N | Y | Y | Y |
| GRF         | Y | Y | Y | N | Y | Y | Y | N | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Hanabi      | Y | Y | Y | N | Y | Y | Y | N | Y | Y | Y | Y | Y | N | N | N | N | N |

You can find a comprehensive list of existing MARL algorithms in different environments  [here](https://github.com/Replicable-MARL/MARLlib/tree/main/envs).

## Installation

Install Ray 
```
pip install ray==1.8.0 # version sensitive
```


Add patch of MARLlib
```
cd patch
python add_patch.py
```

**Y** to replace source-packages code

**Attention**: Above is the common installation. Each environment needs extra dependency. Please read the installation instruction in envs/base_env/install.

## Examples

```
python marl/main.py --algo_config=MAPPO [--finetuned] --env-config=smac with env_args.map_name=3m
```
--finetuned is optional, force using the finetuned hyperparameter 

## Navigation

We provide an introduction to the code directory to help you get familiar with the codebase:

* top level directory structure

<div align="center">
<img src=image/code-MARLlib.png width=120% />
</div>

This picture is in image/code-MARLlib.png

* MARL directory structure

<div align="center">
<img src=image/code-MARL.png width=70% />
</div>

This picture is in image/code-MARL.png.png

* ENVS directory structure

<div align="center">
<img src=image/code-ENVS.png width=70% />
</div>

This picture is in image/code-ENVS.png.png

## Contribute

MARLlib is friendly to incorporating a new environment. Besides the ten we already implemented, we support almost all kinds of MARL environments.
Before contributing new environment, you must know:

Things you ought to cover:

- provide a new environment interface python file, follow the style of [MARLlib/envs/base_env](https://github.com/Replicable-MARL/MARLlib/tree/main/envs/base_env)
- provide a corresponding config yaml file, follow the style of [MARLlib/envs/base_env/config](https://github.com/Replicable-MARL/MARLlib/tree/main/envs/base_env/config)
- provide a corresponding instruction readme file, follow the style of [MARLlib/envs/base_env/install](https://github.com/Replicable-MARL/MARLlib/tree/main/envs/base_env/install)

Things not essential:

- change the MARLlib data processing pipeline
- provide a unique runner or controller specific to the environment 
- concern about the data logging 

The ten environments we already contained have covered great diversity in action space,  observation space, agent-env interaction style, task mode, additional information like action mask, etc. 
The best practice to incorporate your environment is to find an existing similar one and provide the same interface.

## Bug Shooting

Our patch files fix most RLlib-related errors on MARL.

Here we only list the common bugs, not RLlib-related. (Mostly is your mistake)

- *observation/action out of space* bug:
    - make sure the observation/action space defined in env init function 
        - has same data type with env returned data (e.g., float32/64)
        - env returned data range is in the space scope (e.g., box(-2,2))
    - the returned env observation contained the required key (e.g., action_mask/state)
    
- *Action NaN is invaild* bug
    - this is common bug espectially in continues control problem, carefully finetune the algorithm's hyperparameter
        - smaller learning rate
        - set some action value bound

    
