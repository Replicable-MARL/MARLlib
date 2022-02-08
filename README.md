#  White Paper for [MARL in One](https://github.com/Theohhhu/[SMAC](HTTPS://GITHUB.COM/OXWHIRL/SMAC)_Ray) Benchmark
-- multi-agent tasks and baselines under a unified framework

### Part I. Overview

We collected most of the existing multi-agent environment and multi-agent reinforcement learning algorithms with different model architecture under one framework based on [**Ray**](https://github.com/ray-project/ray)'s [**RLlib**](https://github.com/ray-project/ray/tree/master/rllib) to boost the MARL research. 

The code is simple and easy to understand. The most common MARL baselines include **independence learning (IQL, IA2C, IPPO, IPG)**, **centralized critic learning (MAPPO, MAA2C)** and **joint Q learning (QMIX, VDN)** are all implemented as simple as possible. The same algorithm under different tasks can be a bit different to satisfy the various settings of multi-agent systems. But once you are familiar with one task, it will be easy to move to another.

We hope everyone in the MARL research community can be benefited from this benchmark.

The basic structure of the repository. Here we take **[SMAC](HTTPS://GITHUB.COM/OXWHIRL/SMAC)** task as an example (name may be slightly different)

```
/
└───SMAC
        └───env     [**wrap the original env to make it compatible with RLLIB**]
                └───starcraft2_rllib.py
                └───starcraft2_rllib_qmix.py
        └───model   [**agent architecture, cc represents for centralized critic**]
                └───gru.py
                └───gru_cc.py
                └───lstm.py
                └───lstm_cc.py
                └───updet.py
                └───updet_cc.py
                └───qmix.py
                └───r2d2.py
        └───utils   [**centralized critic tools**]
                └───mappo_tools.py
                └───maa2c_tools.py
        └───metrics [**customized metrics for logging**]
                └───SMAC_callback.py
        └───README.md
        └───run_env.py
        └───config_env.py 

```

All codes are built based on RLLIB with some necessary extensions. The tutorial of RLLIB can be found in https://docs.ray.io/en/latest/rllib/index.html.
Fast examples can be found in https://docs.ray.io/en/latest/rllib-examples.html. 
These will help you dive into the basic usage of RLLIB.

### Part II. Environment

##### I. Motivation

RL Community has boomed thanks to some great works like [**OpenAI**](https://openai.com/)'s [**Gym**](https://github.com/openai/gym) and [**RLLIB**](https://github.com/ray-project/ray/tree/master/rllib) under [**Anyscale**](https://www.anyscale.com/)'s [**Ray**](https://github.com/ray-project/ray) framework. **Gym** provides a unified style of RL environment interface. **RLLIB** makes the RL training more scalable and efficient.

But unlike the single-agent RL tasks, multi-agent RL have its unique challenge. 
For example, if you look into the task settings of following multi-agent tasks, you will find so many differences in...

- **Game Process**: Turn-based (Go, [Hanabi](https://github.com/deepmind/hanabi-learning-environment))/ Simultaneously ([StarcarftII]( https://github.com/deepmind/pysc2), [Pommerman](https://github.com/MultiAgentLearning/playground))
-  **Learning Mode**: Cooperative ([SMAC](HTTPS://GITHUB.COM/OXWHIRL/SMAC)) / Collaborative ([RWARE](https://github.com/semitable/robotic-warehouse)) / Competitve ([Neural-MMO](https://github.com/NeuralMMO/environment)) / Mixed ([Pommerman](https://github.com/MultiAgentLearning/playground))
-  **Observability**: Full observation ([Pommerman](https://github.com/MultiAgentLearning/playground)) / Partial observation ([SMAC](https://github.com/oxwhirl/smac))
-  **Action Space**: Discrete ([SMAC](https://github.com/oxwhirl/smac)) / Continuous  ([MPE](https://github.com/openai/multiagent-particle-envs)) / Mixed ([Derk's Gym](https://gym.derkgame.com/)) / Multi-Discrete ([Neural-MMO](https://github.com/NeuralMMO/environment))
-  **Action Mask**: Provided ([Hanabi](https://github.com/deepmind/hanabi-learning-environment), [SMAC](https://github.com/oxwhirl/smac)) / None ([MPE](https://github.com/openai/multiagent-particle-envs), [Pommerman](https://github.com/MultiAgentLearning/playground))
-  **Global State**: Provided ([SMAC](https://github.com/oxwhirl/smac)) / None ([Neural-MMO](https://github.com/NeuralMMO/environment))

So It is nearly impossible for one MARL algorithm to be directly applied to all MARL tasks without some adjustment. Algorithms proposed with papers titled "xxxx MARL xxxx" always claim their model performs better than existing. But given these significantly different settings of multi-agent tasks, many of them can only cover a very limited type of multi-agent tasks or are they based on settings different from the standard one.  

From this perspective, the way of fairly comparing the performance of different MARL algorithms should be treated much more seriously.  We will discuss this later in **Part III. Algorithms**.

##### II. What we did in this benchmark

In this benchmark, we incorporate the characteristics of both Gym and RLLIB.

All the environments we collected are modified (if necessary) to provide Gym style interface for multi-agent interaction. The unified environments are then registered to RAY and should be RLLIB compatible. Algorithms like joint Q learning (QMIX, VDN) need a conforming return value for the step function. We have provided them for cooperative/collaborative multi-agent tasks like [Google-Football](https://github.com/google-research/football) and [Pommerman](https://github.com/MultiAgentLearning/playground).

We make full use of RLLIB to provide a standard but extendable MARL training pipeline. All MARL algorithms (like Centralized Critic strategy) and agent's "brain" (Neural Network like CNN + RNN) can be easily customized under RLLIB's API. 

Some multi-agent tasks may cost weeks of training. But thanks to RAY, the training process can be easily paralleled with only slight configure file modification. RAY is good at allocating your computing resources for the best training/sampling efficiency.

##### III. Supported Multi-agent System / Tasks

Most of the popular environment in MARL research has been incorporated in this benchmark:

| Env Name | Learning Mode | Observability | Action Space | Observations |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| [LBF](https://github.com/semitable/lb-foraging)  | Mixed | Both | Discrete | Discrete |
| [RWARE](https://github.com/semitable/robotic-warehouse)  | Collaborative | Partial | Discrete | Discrete |
| [MPE](https://github.com/openai/multiagent-particle-envs)  | Mixed | Both | Both | Continuous |
| [SMAC](https://github.com/oxwhirl/smac)  | Cooperative | Partial | Discrete | Continuous |
| [Neural-MMO](https://github.com/NeuralMMO/environment)  | Competitive | Partial | Multi-Discrete | Continuous |
| [Meta-Drive](https://github.com/decisionforce/metadrive)  | Collaborative | Partial | Continuous | Continuous |
| [Pommerman](https://github.com/MultiAgentLearning/playground)  | Mixed | Both | Discrete | Discrete |
| [Google-Football](https://github.com/google-research/football)  | Collaborative | Full | Discrete | Continuous |
| [Derk's Gym](https://gym.derkgame.com/)  | Mixed | Partial | Mixed | Continuous |
| [Hanabi](https://github.com/deepmind/hanabi-learning-environment) | Cooperative | Partial | Discrete | Discrete |


Each environment has a top directory. No inter-dependency exists between two environments. If you are interested in one environment, all the related code is there, in one directory. 

Each environment has a readme file, standing as the instruction for this task, talking about env settings, installation, supported algorithms, important notes, bugs shooting, etc.

![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+) **TODO** We provide original environment link. and a list of algorithms that can be used in this environment. The detailed content of this part can be found in  **Part III. Algorithms**.

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
- [LBF](https://github.com/semitable/lb-foraging)
- [RWARE](https://github.com/semitable/robotic-warehouse)
- [MPE]( https://github.com/openai/multiagent-particle-envs)
- [SMAC]( https://github.com/oxwhirl/smac)
- [Neural-MMO](https://github.com/NeuralMMO/environment) (CompetitionR1)
- [Meta-Drive](https://github.com/decisionforce/metadrive)
- [Pommerman](https://github.com/MultiAgentLearning/playground)
- [Derk's Gym](https://gym.derkgame.com/)
- [Google-Football](https://github.com/google-research/football)
- [Hanabi](https://github.com/deepmind/hanabi-learning-environment)
