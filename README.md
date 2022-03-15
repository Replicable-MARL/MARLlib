#  White Paper for [MARL in One](https://github.com/Theohhhu/[SMAC](HTTPS://GITHUB.COM/OXWHIRL/SMAC)_Ray) Benchmark
-- multi-agent tasks and baselines under a unified framework

### TODO

**basic algorithms**
- [x] COMA 
- [x] DDPG
- [x] VDAC 
- [x] VDPPO


**extensions**
- [ ] Grouping
- [ ] Message sending
- [ ] Modelling others
- [ ] Multi-task Learning

**(optional) MARL environments**
- [ ] Multi-agent Mujoco
- [ ] Overcooked-AI
- [ ] MAgent
- [ ] Go-Bigger






### Part I. Overview

We collected most of the existing multi-agent environment and multi-agent reinforcement learning algorithms with different model architecture under one framework based on [**Ray**](https://github.com/ray-project/ray)'s [**RLlib**](https://github.com/ray-project/ray/tree/master/rllib) to boost the MARL research. 

The code is simple and easy to understand. The most common MARL baselines include **independence learning (IQL, IA2C, IPPO, IPG)**, **centralized critic learning (COMA, MADDPG, MAPPO, MAA2C)** and **value decomposition (QMIX, VDN, VDA2C, VDPPO)** are all implemented as simple as possible. The same algorithm under different tasks can be a bit different to satisfy the various settings of multi-agent systems. But once you are familiar with one task, it will be easy to move to another.

We hope everyone in the MARL research community can be benefited from this benchmark.

The basic structure of the repository. Here we take **[SMAC](HTTPS://GITHUB.COM/OXWHIRL/SMAC)** task as an example (name may be slightly different)

```
/
└───SMAC
        └───env     [**wrap the original env to make it compatible with RLLIB**]
                └───starcraft2_rllib.py
                
        └───model   [**architecture, cc represents for centralized critic**]
                └───gru.py
                └───gru_cc.py
                
        └───utils   [**algorithm**]
                └───mappo_tools.py
                └───vda2c_tools.py
                
        └───metrics [**customized metrics for logging**]
                └───callback.py
                
        └───README.md
        └───run_env.py
        └───config_env.py 

```

All codes are built based on RLLIB with some necessary extensions. The tutorial of RLLIB can be found in https://docs.ray.io/en/latest/rllib/index.html.
Fast examples can be found in https://docs.ray.io/en/latest/rllib-examples.html. 
These will help you easily dive into RLLIB.

The following parts are organized as:
- **Part II.Environment**: available environments with the description
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

All the environments we collected are modified (if necessary) to provide Gym style interface for multi-agent interaction. The unified environments are then registered to RAY and should be RLLIB compatible. Algorithms like Value Decomposition (QMIX, VDN, VDA2C, VDPPO) need a conforming return value for the step function. We have provided them for cooperative/collaborative multi-agent tasks like [Google-Football](https://github.com/google-research/football) and [Pommerman](https://github.com/MultiAgentLearning/playground).

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

| Algorithm | Learning Mode | Need Global State | Action | Learning Mode  | Type |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| IQL  | Mixed | No | Discrete | Independent Learning | Off Policy
| IPG  | Mixed | No | Both | Independent Learning | On Policy
| IAC  | Mixed | No | Both | Independent Learning | On Policy
| IDDPG  | Mixed | No | Continuous | Independent Learning | Off Policy
| IPPO  | Mixed | No | Both | Independent Learning | On Policy
| COMA  | Mixed | No | Both | Centralized Critic | On Policy
| MADDPG  | Mixed | Better | Continuous | Centralized Critic | Off Policy
| MAAC  | Mixed | Better | Both | Centralized Critic | On Policy
| MAPPO  | Mixed | Better | Both | Centralized Critic | On Policy
| VDN | Cooperative | No | Discrete | Value Decomposition | Off Policy
| QMIX  | Cooperative | Yes | Discrete | Value Decomposition | Off Policy
| VDAC  | Cooperative | Better | Both | Value Decomposition | On Policy
| VDPPO | Cooperative | Better | Both | Value Decomposition | On Policy

**Current Task & Available algorithm map**: Y for support, N for unavailable, P for partly available
(Note: in our code, independent algorithms may not have **I** as prefix. For instance, PPO = IPPO)

| Env w Algorithm | IQL(R2D2) | IPG | IAC | IDDPG | IPPO | COMA | MADDPG | MAAC | MAPPO | VDN | QMIX | VDAC | VDPPO 
| --------- | -------- | -------- | -------- | -------- | -------- | -------- |--------- | -------- | -------- | -------- | -------- | -------- | -------- |
| LBF         | Y | Y | Y | Y | Y | Y | N | Y | Y | P | P | P | P |
| RWARE       | Y | Y | Y | Y | Y | Y | N | Y | Y | Y | Y | Y | Y |
| MPE         | P | Y | Y | P | Y | P | P | Y | Y | N | N | P | P |
| SMAC        | Y | Y | Y | Y | Y | Y | N | Y | Y | Y | Y | Y | Y |
| Meta-Drive  | N | Y | Y | Y | Y | N | Y | Y | Y | N | N | N | N |
| Pommerman   | Y | Y | Y | Y | Y | P | N | P | N | P | P | N | N |
| GRF         | Y | Y | Y | Y | Y | N | N | N | N | Y | Y | Y | Y |
| Derk's Gym  | N | Y | Y | N | Y | N | N | Y | Y | N | N | N | N |
| Hanabi      | Y | Y | Y | N | Y | Y | N | Y | Y | N | N | N | N |
| Neural-MMO  | N | Y | Y | N | Y | N | N | Y | N | N | N | N | N |

**Current Task & Neural Arch map**: Y for support, N for unavailable

| Env w Arch | MLP | GRU | LSTM | CNN | Transformer | 
| --------- | -------- | -------- | -------- | -------- | -------- |
| LBF  | N | Y | Y | N | N |
| RWARE  | N | Y | Y | N | N | 
| MPE  | Y | Y | Y | N | N | 
| SMAC  | N | Y | Y | N | Y | 
| Neural-MMO  | N | Y | Y | Y | Y |
| Meta-Drive  | N | Y | Y | Y | Y | 
| Pommerman  | N | Y | Y | Y | N |
| Google-Football  | Y | Y | Y | Y | Y | 
| Derk's Gym  | N | Y | Y | N | N |
| Hanabi  | N | Y | Y | N | N |


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
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf) **[B][2017]**
- [Learning attentional communication for multi-agent cooperation](https://proceedings.neurips.cc/paper/2018/file/6a8018b3a00b69c008601b8becae392b-Paper.pdf) **[S][2018]**
- [learning when to communicate at scale in multiagent cooperative and competitive tasks](https://arxiv.org/pdf/1812.09755) **[S][2018]**
- [Qtran: Learning to factorize with transformation for cooperative multi-agent reinforcement learning](http://proceedings.mlr.press/v97/son19a/son19a.pdf) **[B][2019]**
- [Robust multi-agent reinforcement learning via minimax deep deterministic policy gradient](https://ojs.aaai.org/index.php/AAAI/article/view/4327/4205) **[R][2019]**
- [Tarmac: Targeted multi-agent communication](http://proceedings.mlr.press/v97/das19a/das19a.pdf) **[S][2019]**
- [Learning Individually Inferred Communication for Multi-Agent Cooperation](https://proceedings.neurips.cc/paper/2020/file/fb2fcd534b0ff3bbed73cc51df620323-Paper.pdf) **[S][2020]**
- [Multi-Agent Game Abstraction via Graph Attention Neural Network](https://ojs.aaai.org/index.php/AAAI/article/view/6211/6067) **[G+S][2020]**
- [Promoting Coordination through Policy Regularization in Multi-Agent Deep Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/file/b628386c9b92481fab68fbf284bd6a64-Paper.pdf) **[E][2020]**
- [Robust Multi-Agent Reinforcement Learning with Model Uncertainty](https://proceedings.neurips.cc/paper/2020/file/774412967f19ea61d448977ad9749078-Paper.pdf) **[R][2020]**
- [Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/file/7967cc8e3ab559e68cc944c44b1cf3e8-Paper.pdf) **[B][2020]**
- [Weighted QMIX Expanding Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf) **[B][2020]**
- [Cooperative Exploration for Multi-Agent Deep Reinforcement Learning](http://proceedings.mlr.press/v139/liu21j/liu21j.pdf) **[E][2021]**
- [Multiagent Adversarial Collaborative Learning via Mean-Field Theory](https://ieeexplore.ieee.org/iel7/6221036/9568742/09238422.pdf?casa_token=43-7BP8rsWgAAAAA:ESpZx5Nunchu6Un6vIaVljiJQrSj7tYGWVgx1x3tGvCMkSktx55ZCopEW8VC4SwfjX6RU_KT_c8) **[R][2021]**
- [The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games](https://arxiv.org/pdf/2103.01955?ref=https://githubhelp.com) **[B][2021]**

#### **SMAC**
- [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/pdf/1706.05296?ref=https://githubhelp.com) **[B][2017]**
- [Counterfactual Multi-Agent Policy Gradients](https://ojs.aaai.org/index.php/AAAI/article/download/11794/11653) **[B][2018]**
- [Multi-Agent Common Knowledge Reinforcement Learning](https://proceedings.neurips.cc/paper/2019/file/f968fdc88852a4a3a27a81fe3f57bfc5-Paper.pdf) **[RG+S][2018]**
- [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf) **[B][2018]**
- [Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control](https://proceedings.neurips.cc/paper/2019/file/14cfdb59b5bda1fc245aadae15b1984a-Paper.pdf) **[S][2019]**
- [Exploration with Unreliable Intrinsic Reward in Multi-Agent Reinforcement Learning](https://arxiv.org/pdf/1906.02138) **[P+E][2019]**
- [Learning nearly decomposable value functions via communication minimization](https://arxiv.org/pdf/1910.05366) **[S][2019]**
- [Liir: Learning individual intrinsic reward in multi-agent reinforcement learning](https://proceedings.neurips.cc/paper/2019/file/07a9d3fed4c5ea6b17e80258dee231fa-Paper.pdf) **[P][2019]**
- [MAVEN: Multi-Agent Variational Exploration](https://proceedings.neurips.cc/paper/2019/file/f816dc0acface7498e10496222e9db10-Paper.pdf) **[E][2019]**
- [Adaptive learning A new decentralized reinforcement learning approach for cooperative multiagent systems](https://ieeexplore.ieee.org/iel7/6287639/8948470/09102277.pdf) **[B][2020]**
- [Counterfactual Multi-Agent Reinforcement Learning with Graph Convolution Communication](https://arxiv.org/pdf/2004.00470) **[S+G][2020]**
- [Deep implicit coordination graphs for multi-agent reinforcement learning](https://arxiv.org/pdf/2006.11438) **[G][2020]**
- [DOP: Off-policy multi-agent decomposed policy gradients](https://openreview.net/pdf?id=6FqKiVAdI3Y) **[B][2020]**
- [F2a2: Flexible fully-decentralized approximate actor-critic for cooperative multi-agent reinforcement learning](https://arxiv.org/pdf/2004.11145) **[B][2020]**
- [From few to more Large-scale dynamic multiagent curriculum learning](https://ojs.aaai.org/index.php/AAAI/article/view/6221/6083) **[MT][2020]**
- [Learning structured communication for multi-agent reinforcement learning](https://arxiv.org/pdf/2002.04235) **[S+G][2020]**
- [Learning efficient multi-agent communication: An information bottleneck approach](http://proceedings.mlr.press/v119/wang20i/wang20i.pdf) **[S][2020]**
- [On the robustness of cooperative multi-agent reinforcement learning](https://ieeexplore.ieee.org/iel7/9283745/9283819/09283830.pdf?casa_token=k2lORHebFEUAAAAA:kmTJ2M4Q67hwRz8fh6LhgoXgwZLPy_idCgBmXDxBjzcJBgnYuLmCc7iDS8KTjbVcRPmal-jV9sM) **[R][2020]**
- [Qatten: A general framework for cooperative multiagent reinforcement learning](https://arxiv.org/pdf/2002.03939) **[B][2020]**
- [Revisiting parameter sharing in multi-agent deep reinforcement learning](https://arxiv.org/pdf/2005.13625) **[RG][2020]**
- [Qplex: Duplex dueling multi-agent q-learning](https://arxiv.org/pdf/2008.01062) **[B][2020]**
- [ROMA: Multi-Agent Reinforcement Learning with Emergent Roles](https://arxiv.org/pdf/2003.08039) **[RG][2020]**
- [Towards Understanding Cooperative Multi-Agent Q-Learning with Value Factorization](https://proceedings.neurips.cc/paper/2021/file/f3f1fa1e4348bfbebdeee8c80a04c3b9-Paper.pdf) **[B][2021]**
- [Contrasting centralized and decentralized critics in multi-agent reinforcement learning](https://arxiv.org/pdf/2102.04402) **[B][2021]**
- [Learning in nonzero-sum stochastic games with potentials](http://proceedings.mlr.press/v139/mguni21a/mguni21a.pdf) **[B][2021]**
- [Natural emergence of heterogeneous strategies in artificially intelligent competitive teams](https://arxiv.org/pdf/2007.03102) **[S+G][2021]**
- [Rode: Learning roles to decompose multi-agent tasks](https://arxiv.org/pdf/2010.01523?ref=https://githubhelp.com) **[RG][2021]**
- [SMIX(λ): Enhancing Centralized Value Functions for Cooperative Multiagent Reinforcement Learning](https://ieeexplore.ieee.org/iel7/5962385/6104215/09466372.pdf?casa_token=TdedVHwLvL4AAAAA:kGSnPCM1wQMte1gloaEBUhgD9kUP1FA3mf1TZ931e7W1RqFAr0ewePlhHkEEEArHva6SikWDFA4) **[B][2021]**
- [Tesseract: Tensorised Actors for Multi-Agent Reinforcement Learning](http://proceedings.mlr.press/v139/mahajan21a/mahajan21a.pdf) **[B][2021]**
- [The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games](https://arxiv.org/pdf/2103.01955?ref=https://githubhelp.com) **[B][2021]**
- [UPDeT: Universal Multi-agent Reinforcement Learning via Policy Decoupling with Transformers](https://openreview.net/pdf?id=v9c7hr9ADKx) **[MT][2021]**
- [Randomized Entity-wise Factorization for Multi-Agent Reinforcement Learning](http://proceedings.mlr.press/v139/iqbal21a/iqbal21a.pdf) **[MT][2021]**
- [Cooperative Multi-Agent Transfer Learning with Level-Adaptive Credit Assignment](https://arxiv.org/pdf/2106.00517?ref=https://githubhelp.com) **[MT][2021]**
- [Uneven: Universal value exploration for multi-agent reinforcement learning](http://proceedings.mlr.press/v139/gupta21a/gupta21a.pdf) **[B][2021]**
- [Value-decomposition multi-agent actor-critics](https://www.aaai.org/AAAI21Papers/AAAI-2412.SuJ.pdf) **[B][2021]**


#### **Pommerman**
- [Using Monte Carlo Tree Search as a Demonstrator within Asynchronous Deep RL](https://arxiv.org/pdf/1812.00045) **[I+T][2018]**
- [Accelerating Training in Pommerman with Imitation and Reinforcement Learning](https://arxiv.org/pdf/1911.04947) **[I][2019]**
- [Agent Modeling as Auxiliary Task for Deep Reinforcement Learning](https://ojs.aaai.org/index.php/AIIDE/article/download/5221/5077/) **[S][2019]**
- [Backplay: man muss immer umkehren](https://arxiv.org/pdf/1807.06919.pdf%20http://arxiv.org/abs/1807.06919) **[I][2019]**
- [Terminal Prediction as an Auxiliary Task for Deep Reinforcement Learning](https://ojs.aaai.org/index.php/AIIDE/article/download/5222/5078) **[B][2019]**
- [Adversarial Soft Advantage Fitting Imitation Learning without Policy Optimization](https://proceedings.neurips.cc/paper/2020/file/9161ab7a1b61012c4c303f10b4c16b2c-Paper.pdf) **[B][2020]**
- [Evolutionary Reinforcement Learning for Sample-Efficient Multiagent Coordination](http://proceedings.mlr.press/v119/majumdar20a/majumdar20a.pdf) **[B][2020]**


#### **Hanabi**
- [Bayesian Action Decoder for Deep Multi-Agent Reinforcement Learning](http://proceedings.mlr.press/v97/foerster19a/foerster19a.pdf) **[B][2019]**
- [Re-determinizing MCTS in Hanabi](https://ieeexplore.ieee.org/iel7/8844551/8847948/08848097.pdf?casa_token=nZ3ZAeyS1-kAAAAA:3FBwAb2lMlQ_ClJIlycoVsensDQFE0pqMeQ8PvMc15Bzoam9inGlWBJmT6D9bKjF1WUL7k5IkS0) **[S+T][2019]**
- [Diverse Agents for Ad-Hoc Cooperation in Hanabi](https://ieeexplore.ieee.org/iel7/8844551/8847948/08847944.pdf?casa_token=oDFhRxwd0XIAAAAA:Vq6oBEA6fotbST9N-RkThJjY5URVVvnwQ8Y0mt1JiD9uLXmXMxt7k8Dqt-VghWJzK8fOgdXFbH0) **[B][2019]**
- [Joint Policy Search for Multi-agent Collaboration with Imperfect Information](https://proceedings.neurips.cc/paper/2020/file/e64f346817ce0c93d7166546ac8ce683-Paper.pdf) **[T][20209]**
- [Off-Belief Learning](http://proceedings.mlr.press/v139/hu21c/hu21c.pdf) **[B][2021]**
- [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/pdf/2103.01955?ref=https://githubhelp.com) **[B][2021]**
- [2021 Trajectory Diversity for Zero-Shot Coordination](http://proceedings.mlr.press/v139/lupu21a/lupu21a.pdf) **[B][2021]**

#### **GRF**
- [Adaptive Inner-reward Shaping in Sparse Reward Games](https://ieeexplore.ieee.org/iel7/9200848/9206590/09207302.pdf?casa_token=T6Xp9_s07OwAAAAA:ECy-wfIOoMq60Mkk3qfitWlSzslNTC5mBkHtVLu1SmJ9STDErl7OYjoptRKU6PMsqh7_4cbP6Jk) **[P][2020]**
- [Factored action spaces in deep reinforcement learning](https://openreview.net/pdf?id=naSAkn2Xo46) **[B][2021]**
- [Semantic Tracklets An Object-Centric Representation for Visual Multi-Agent Reinforcement Learning](https://ieeexplore.ieee.org/iel7/9635848/9635849/09636592.pdf?casa_token=x8RsQf74KUUAAAAA:lp6vsCBIaMlYbhP4xoIM2279USMn3-KW73DxyhejGOz-hiG2kDRqQIrNSABy6IlAYdU4BvRqAnc) **[B][2021]**
- [TiKick: Towards Playing Multi-agent Football Full Games from Single-agent Demonstrations](https://arxiv.org/pdf/2110.04507) **[F][2021]**


#### **LBF & RWARE**
- [Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/file/7967cc8e3ab559e68cc944c44b1cf3e8-Paper.pdf) **[B][2020]**
- [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://arxiv.org/pdf/2006.07869) **[B][2021]**
- [Learning Altruistic Behaviors in Reinforcement Learning without External Rewards](https://arxiv.org/pdf/2107.09598) **[B][2021]**
- [Scaling Multi-Agent Reinforcement Learning with Selective Parameter Sharing](http://proceedings.mlr.press/v139/christianos21a/christianos21a.pdf) **[RG][2021]**

#### **MetaDrive**
- [Learning to Simulate Self-Driven Particles System with Coordinated Policy Optimization](https://proceedings.neurips.cc/paper/2021/file/594ca7adb3277c51a998252e2d4c906e-Paper.pdf) **[B][2021]**
- [Safe Driving via Expert Guided Policy Optimization](https://proceedings.mlr.press/v164/peng22a/peng22a.pdf) **[I][2021]**

#### **NeuralMMO**
- [Neural MMO: A Massively Multiagent Game Environment for Training and Evaluating Intelligent Agents](https://arxiv.org/pdf/1903.00784) **[B][2019]**
- [The Neural MMO Platform for Massively Multiagent Research](https://arxiv.org/pdf/2110.07594) **[B][2021]**


**(Note: this is not a comprehensive list. Only representative papers are selected.)**

### **Part V. Extensions**

- **Grouping / Parameter Sharing**:
  - Fully sharing (All in one group)
  - Selectively sharing (Several groups)
  - No sharing (No group)

- **Communication / Information Exchange**:
  - Sending Message (Explicit communication)
  - Modeling Others (Predict teammates/opponents action)

- **Multi-task Learning / Transfer Learning / Continue learning**
  - RNN
  - Transformer
  - 

### **Part VI. Bug Shooting**
- ppo related bug: refer to https://github.com/ray-project/ray/pull/20743. 
  - make sure sgd_minibatch_size > max_seq_len
  - enlarge the sgd_minibatch_size (128 in default)

## Acknowledgement


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
