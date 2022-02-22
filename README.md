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
These will help you easily dive into RLLIB.

The following parts are organized as:
- **Part II. Environment**: available environments with the description
- **Part III. Baseline Algorithms**: implemented baseline MARL algorithms cover **independent learning / centralized critic / value decomposition**
- **Part IV. State of the Art**: existing works on the environments we provide, with topic annotation.
- **Part V. Extensions**: general module that can be applied to most of the environments/baseline algorithms

### Part II. Environment

#### Motivation

RL Community has boomed thanks to some great works like [**OpenAI**](https://openai.com/)'s [**Gym**](https://github.com/openai/gym) and [**RLLIB**](https://github.com/ray-project/ray/tree/master/rllib) under [**Anyscale**](https://www.anyscale.com/)'s [**Ray**](https://github.com/ray-project/ray) framework. **Gym** provides a unified style of RL environment interface. **RLLIB** makes the RL training more scalable and efficient.

But unlike the single-agent RL tasks, multi-agent RL has its unique challenge. 
For example, if you look into the task settings of the following multi-agent tasks, you will find so many differences in...

- **Game Process**: Turn-based (Go, [Hanabi](https://github.com/deepmind/hanabi-learning-environment))/ Simultaneously ([StarcarftII]( https://github.com/deepmind/pysc2), [Pommerman](https://github.com/MultiAgentLearning/playground))
-  **Learning Mode**: Cooperative ([SMAC](HTTPS://GITHUB.COM/OXWHIRL/SMAC)) / Collaborative ([RWARE](https://github.com/semitable/robotic-warehouse)) / Competitve ([Neural-MMO](https://github.com/NeuralMMO/environment)) / Mixed ([Pommerman](https://github.com/MultiAgentLearning/playground))
-  **Observability**: Full observation ([Pommerman](https://github.com/MultiAgentLearning/playground)) / Partial observation ([SMAC](https://github.com/oxwhirl/smac))
-  **Action Space**: Discrete ([SMAC](https://github.com/oxwhirl/smac)) / Continuous  ([MPE](https://github.com/openai/multiagent-particle-envs)) / Mixed ([Derk's Gym](https://gym.derkgame.com/)) / Multi-Discrete ([Neural-MMO](https://github.com/NeuralMMO/environment))
-  **Action Mask**: Provided ([Hanabi](https://github.com/deepmind/hanabi-learning-environment), [SMAC](https://github.com/oxwhirl/smac)) / None ([MPE](https://github.com/openai/multiagent-particle-envs), [Pommerman](https://github.com/MultiAgentLearning/playground))
-  **Global State**: Provided ([SMAC](https://github.com/oxwhirl/smac)) / None ([Neural-MMO](https://github.com/NeuralMMO/environment))

So It is nearly impossible for one MARL algorithm to be directly applied to all MARL tasks without some adjustment. Algorithms proposed with papers titled "xxxx MARL xxxx" always claim their model performs better than existing. But given these significantly different settings of multi-agent tasks, many of them can only cover a very limited type of multi-agent tasks, or are they based on settings different from the standard one.  

From this perspective, the way of fairly comparing the performance of different MARL algorithms should be treated much more seriously.  We will discuss this later in **Part III. Algorithms**.

#### What we did in this benchmark

In this benchmark, we incorporate the characteristics of both Gym and RLLIB.

All the environments we collected are modified (if necessary) to provide Gym style interface for multi-agent interaction. The unified environments are then registered to RAY and should be RLLIB compatible. Algorithms like joint Q learning (QMIX, VDN) need a conforming return value for the step function. We have provided them for cooperative/collaborative multi-agent tasks like [Google-Football](https://github.com/google-research/football) and [Pommerman](https://github.com/MultiAgentLearning/playground).

We make full use of RLLIB to provide a standard but extendable MARL training pipeline. All MARL algorithms (like Centralized Critic strategy) and agent's "brain" (Neural Network like CNN + RNN) can be easily customized under RLLIB's API. 

Some multi-agent tasks may cost weeks of training. But thanks to RAY, the training process can be easily paralleled with only slight configure file modification. RAY is good at allocating your computing resources for the best training/sampling efficiency.

#### Supported Multi-agent System / Tasks

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

### Part III. Baseline Algorithms

We provide three types of MARL algorithms as our baselines including:

**Independent Learning:** 
IQL
IPG
IAC
IDDPG
IPPO

**Centralized Critic:**
COMA 
MADDPG 
MAAC 
MAPPO

**Value Decomposition:**
VDN
QMIX
VDAC
VDPPO

Here is a chart describing the characteristics of each algorithm:

| Algorithm | Learning Mode | Need Global State | Action | Type |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| IQL  | Mixed | No | Discrete | Independent Learning |
| IPG  | Mixed | No | Both | Independent Learning |
| IAC  | Mixed | No | Both | Independent Learning |
| IDDPG  | Mixed | No | Continuous | Independent Learning |
| IPPO  | Mixed | No | Both | Independent Learning |
| COMA  | Mixed | No | Both | Centralized Critic |
| MADDPG  | Mixed | No | Continuous | Centralized Critic |
| MAAC  | Mixed | No | Both | Centralized Critic |
| MAPPO  | Mixed | No | Both | Centralized Critic |
| VDN | Cooperative | No | Discrete | Value Decomposition |
| QMIX  | Cooperative | Yes | Discrete | Value Decomposition |
| VDAC  | Cooperative | Yes | Both | Value Decomposition |
| VDPPO | Cooperative | Yes | Both | Value Decomposition |



### Part IV. State of the Art

**[B]** Basic
**[S]** Information Sharing
**[RG]** Behavior/Role Grouping
**[I]** Imitation
**[G]** Graph
**[E]** Exploration
**[R]** Robust
**[P]** Reward Shaping
**[F]** Offline
**[T]** Tree Search
**[MT]** Multi-task

#### **MPE**
- Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments **[B][2017]**
- Learning attentional communication for multi-agent cooperation **[S][2018]**
- learning when to communicate at scale in multiagent cooperative and competitive tasks **[S][2018]**
- Qtran: Learning to factorize with transformation for cooperative multi-agent reinforcement learning **[B][2019]**
- Robust multi-agent reinforcement learning via minimax deep deterministic policy gradient **[R][2019]**
- Tarmac: Targeted multi-agent communication **[S][2019]**
- Learning Individually Inferred Communication for Multi-Agent Cooperation **[S][2020]**
- Multi-Agent Game Abstraction via Graph Attention Neural Network **[G+S][2020]**
- Promoting Coordination through Policy Regularization in Multi-Agent Deep Reinforcement Learning **[E][2020]**
- Robust Multi-Agent Reinforcement Learning with Model Uncertainty **[R][2020]**
- Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning **[B][2020]**
- Weighted QMIX Expanding Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning **[B][2020]**
- Cooperative Exploration for Multi-Agent Deep Reinforcement Learning **[E][2021]**
- Multiagent Adversarial Collaborative Learning via Mean-Field Theory **[R][2021]**
- The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games **[B][2021]**

#### **SMAC**
- Value-Decomposition Networks For Cooperative Multi-Agent Learning **[B][2017]**
- Counterfactual Multi-Agent Policy Gradients **[B][2018]**
- Multi-Agent Common Knowledge Reinforcement Learning **[RG+S][2018]**
- QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning **[B][2018]**
- Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control **[S][2019]**
- Exploration with Unreliable Intrinsic Reward in Multi-Agent Reinforcement Learning **[P+E][2019]**
- Learning nearly decomposable value functions via communication minimization **[S][2019]**
- Liir: Learning individual intrinsic reward in multi-agent reinforcement learning **[P][2019]**
- MAVEN: Multi-Agent Variational Exploration **[E][2019]**
- Adaptive learning A new decentralized reinforcement learning approach for cooperative multiagent systems **[B][2020]**
- Counterfactual Multi-Agent Reinforcement Learning with Graph Convolution Communication **[S+G][2020]**
- Deep implicit coordination graphs for multi-agent reinforcement learning **[G][2020]**
- DOP: Off-policy multi-agent decomposed policy gradients **[B][2020]**
- F2a2: Flexible fully-decentralized approximate actor-critic for cooperative multi-agent reinforcement learning **[B][2020]**
- From few to more Large-scale dynamic multiagent curriculum learning **[MT][2020]**
- Learning structured communication for multi-agent reinforcement learning **[S+G][2020]**
- Learning efficient multi-agent communication: An information bottleneck approach **[S][2020]**
- On the robustness of cooperative multi-agent reinforcement learning **[R][2020]**
- Qatten: A general framework for cooperative multiagent reinforcement learning **[B][2020]**
- Revisiting parameter sharing in multi-agent deep reinforcement learning **[RG][2020]**
- Qplex: Duplex dueling multi-agent q-learning **[B][2020]**
- ROMA: Multi-Agent Reinforcement Learning with Emergent Roles **[RG][2020]**
- Towards Understanding Cooperative Multi-Agent Q-Learning with Value Factorization **[B][2021]**
- Contrasting centralized and decentralized critics in multi-agent reinforcement learning **[B][2021]**
- Learning in nonzero-sum stochastic games with potentials **[B][2021]**
- Natural emergence of heterogeneous strategies in artificially intelligent competitive teams **[S+G][2021]**
- Rode: Learning roles to decompose multi-agent tasks **[RG][2021]**
- SMIX(λ): Enhancing Centralized Value Functions for Cooperative Multiagent Reinforcement Learning **[B][2021]**
- Tesseract: Tensorised Actors for Multi-Agent Reinforcement Learning **[B][2021]**
- The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games **[B][2021]**
- UPDeT: Universal Multi-agent Reinforcement Learning via Policy Decoupling with Transformers **[MT][2021]**
- Randomized Entity-wise Factorization for Multi-Agent Reinforcement Learning **[MT][2021]**
- Cooperative Multi-Agent Transfer Learning with Level-Adaptive Credit Assignment **[MT][2021]**
- Uneven: Universal value exploration for multi-agent reinforcement learning **[B][2021]**
- Value-decomposition multi-agent actor-critics **[B][2021]**


#### **Pommerman**
- Using Monte Carlo Tree Search as a Demonstrator within Asynchronous Deep RL **[I+T][2018]**
- Accelerating Training in Pommerman with Imitation and Reinforcement Learning **[I][2019]**
- Agent Modeling as Auxiliary Task for Deep Reinforcement Learning **[S][2019]**
- Backplay man muss immer umkehren **[I][2019]**
- Terminal Prediction as an Auxiliary Task for Deep Reinforcement Learning **[B][2019]**
- Adversarial Soft Advantage Fitting Imitation Learning without Policy Optimization **[B][2020]**
- Evolutionary Reinforcement Learning for Sample-Efficient Multiagent Coordination **[B][2020]**


#### **Hanabi**
- Bayesian Action Decoder for Deep Multi-Agent Reinforcement Learning **[B][2019]**
- Re-determinizing MCTS in Hanabi **[S+T][2019]**
- Joint Policy Search for Multi-agent Collaboration with Imperfect Information **[T][20209]**
- Off-Belief Learning **[B][2021]**
- The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games **[B][2021]**

#### **GRF**
- Adaptive Inner-reward Shaping in Sparse Reward Games **[P][2020]**
- Factored action spaces in deep reinforcement learning **[B][2021]**
- TiKick: Towards Playing Multi-agent Football Full Games from Single-agent Demonstrations **[F][2021]**


#### **LBF & RWARE**
- Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning **[B][2020]**
- Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks **[B][2021]**
- Learning Altruistic Behaviors in Reinforcement Learning without External Rewards **[B][2021]**
- Scaling Multi-Agent Reinforcement Learning with Selective Parameter Sharing **[RG][2021]**

#### **MetaDrive**
- Learning to Simulate Self-Driven Particles System with Coordinated Policy Optimization **[B][2021]**
- Safe Driving via Expert Guided Policy Optimization **[I][2021]**

#### **NeuralMMO**
- Neural MMO: A Massively Multiagent Game Environment for Training and Evaluating Intelligent Agents **[B][2021]**

**(Note: this is not a comprehensive list. Only representative papers are selected.)**

### **Part V. Extensions**

- **Grouping / Parameter Sharing**:
  - Fully Sharing
  - Partly Sharing (Selectively Sharing)
  - No Sharing

- **Communication / Information Sharing**:
  - Sending Message
  - Modeling Others

- **Multi-task Learning / Transfer Learning**
  - RNN
  - Transformer



----------------------------
# MARL in One Repository
This is fast multi-agent reinforcement learning (MARL) baseline built based on **ray[rllib]**, 
containing most of the popular multi-agent systems (MAS) and providing available MARL algorithms of each environment.

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
