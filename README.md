
 #  White Paper for [MARL in One](https://github.com/Theohhhu/SMAC_Ray) Benchmark
-- multi-agent tasks and baselines under a unified framework

### Part I. Overview

We collected most of the existing multi-agent environment and multi-agent reinforcement learning algorithms with different model architecture under one framework based on Ray's RLlib to boost the MARL research. 

The code is simple and easy to understand. The common MARL baselines **including independence learning (IA2C, IQL, IPPO, IPG)**, **centralized critic learning (MAPPO, MAA2C)** and **joint Q learning (QMIX, VDN)** are all implemented as simple but slightly different to satisfy the various setting of the collected multi-agent systems.

We hope everyone in the MARL research community can be benefited from this benchmark.

The basic structure of the repository. Here take **SMAC** as an example (name may be slightly different)

```
/
└───SMAC
        └───env
                └───starcraft2_rllib.py
                └───starcraft2_rllib_qmix.py
        └───model
                └───gru.py
                └───gru_cc.py
                └───lstm.py
                └───lstm_cc.py
                └───updet.py
                └───updet_cc.py
                └───qmix.py
                └───r2d2.py
        └───utils
                └───mappo_tools.py
                └───maa2c_tools.py
        └───metrics
                └───smac_callback.py
        └───README.md
        └───run_env.py
        └───config_env.py 

```

All codes are built on RLLIB with the necessary extension. The tutorial of rllib can be found in https://docs.ray.io/en/latest/rllib/index.html.
Examples can be found in https://docs.ray.io/en/latest/rllib-examples.html. 
These will help you fast learn the basic usage of RLLIB.

### Part II. Environment

##### I. Motivation

RL Community has boomed thanks to some great works like OpenAI's Gym and RLLIB under Anyscale's RAY framework. Gym provides a unified style of RL environment interface. RLLIB makes the RL training more scalable and efficient.

But unlike the single-agent RL tasks, multi-agent RL tasks various a lot. 

Instance:
- **Game Process**: Turn-based (Go, Hanabi)/ Simultaneously (StarcarftII, Pommerman)
-  **Learning Mode**: Cooperative (SMAC) / Collaborative (RWARE) / Competitve (Neural MMO) / Mixed (Pommerman)
-  **Observability**: Full observation (Pommerman) / Partial observation (SMAC)
-  **Action Space**: Discrete (SMAC) / Continuous  (MPE) / Mixed (Derk's Gym) / Multi-Discrete (Neural MMO)
-  **Action Mask**: Provided (Hanabi, SMAC) / None (MPE, Pommerman)
-  **Global State**: Provided (SMAC) / None (Neural MMO)


Algorithms proposed with papers titled "xxxx MARL xxxx" always claim their model performs better than existing. But given these significantly different attributes of multi-agent systems, many of them can only cover a very limited type of multi-agent tasks or are based on settings different from the standard one.  

From this perspective, the way of fairly comparing the performance of different MARL algorithms should be treated much more seriously.  We will discuss this later in **Part III. Algorithms**.

##### II. What we did in this benchmark


In this benchmark, we corperate the characteristics of both Gym and RLLIB.
Specifically, all the environments we collected are modified (if necessary) to provide Gym style interface for multi-agent interaction. The unified environments are then registered to RAY to be RLLIB compatible. 

We make full use of RLLIB to provide a standard but extendable MARL training pipeline. All MARL algorithms (like Centralized Critic strategy) and agent's "brain" (Neural Network like CNN + RNN) can be easily customized. Environments like Neural MMO may cost weeks of training, but thanks to RAY, the training can be easily paralleled with slight modification on the configuration file.

##### III. Supported Multi-agent System / Tasks

Most of the popular environment in MARL research has been corperated in this benchmark:

| Env Name | Learning Mode | Observability | Action Space | Observations |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| LBF  | Mixed | Both | Discrete | Discrete |
| RWARE  | Collaborative | Partial | Discrete | Discrete |
| MPE  | Mixed | Both | Both | Continuous |
| SMAC  | Cooperative | Partial | Discrete | Continuous |
| Neural-MMO  | Competitive | Partial | Multi-Discrete | Continuous |
| Meta-Drive  | Collaborative | Partial | Continuous | Continuous |
| Pommerman  | Mixed | Both | Discrete | Discrete |
| Google-Football  | Mixed | Full | Discrete | Continuous |
| Derk's Gym  | Mixed | Partial | Mixed | Continuous |
| Hanabi | Cooperative | Partial | Discrete | Discrete |


Each environment has a top directory. No inter-dependency exists between two environments. If you are interested in one environment, all the related code is there, in one directory. 

Each environment has a readme file, standing like instruction on env description, env install, supported algorithms, important notes, bugs shooting, etc.

![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) **TODO** We provide original environment link. Many of them have competition on AIcrowd for you to refer to. We also provide a list of algorithms that can be used in this environment. The detailed content of this part can be found in  **Part III. Algorithms**.

### Part III. Algorithms
![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) **TODO**
1. a large survey about currently available marl algos
    1. spotlight of each algo
    2. Tasks they can cover
    3. some discussion
2. an introduction of algorithms we used in this benchmark
    1. how they work
    2. tasks they can cover
    3. what extensions can be made upon


----------------------------
# MARL in One Repository
This is fast multi-agent reinforcement learning (MARL) baseline built based on **ray[rllib]**, 
containing most of the popular multi-agent system (MAS) and providing available MARL algorithms of each environment.

## Getting started on Linux
Install Ray
> pip install ray==1.8.0 # version is important

Please annotate one line source code to avoid parallel env seed bug (some env need this)
at ray.rllib.evaluation.rollout_worker.py line 508

> _update_env_seed_if_necessary(self.env, seed, worker_index, 0)

Fix Multi-agent + RNN bug according to

https://github.com/ray-project/ray/pull/20743/commits/70861dae9398823814b5b28e3593cc98159c9c44

In **ray/rllib/policy/rnn_sequencing.py** about line 130-150

        for i, k in enumerate(feature_keys_):
            batch[k] = tree.unflatten_as(batch[k], feature_sequences[i])
        for i, k in enumerate(state_keys):
            batch[k] = initial_states[i]
        batch[SampleBatch.SEQ_LENS] = np.array(seq_lens)

        # add two lines here
        if dynamic_max:
            batch.max_seq_len = max(seq_lens)

        if log_once("rnn_ma_feed_dict"):
            logger.info("Padded input for RNN/Attn.Nets/MA:\n\n{}\n".format(
                summarize({
                    "features": feature_sequences,
                    "initial_states": initial_states,
                    "seq_lens": seq_lens,
                    "max_seq_len": max_seq_len,
                })))

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
  
### with neural arch
- GRU
- LSTM
- UPDeT

## current support env
- LBF
- RWARE
- MPE
- SMAC
- Neural-MMO (CompetitionR1)
- Meta-Drive
- Pommerman
- Derk's Gym
- Google-Football
- Hanabi



