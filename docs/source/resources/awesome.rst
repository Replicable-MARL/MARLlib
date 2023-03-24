.. _concept:

***************************************
Awesome Paper List
***************************************

We collect most of the existing MARL algorithms based on the multi-agent environment they choose to conduct on, with tag to annotate the sub-topic.


.. contents::
    :local:
    :depth: 3

[B] Basic [S] Information Sharing [RG] Behavior/Role Grouping [I] Imitation [G] Graph [E] Exploration [R] Robust [P] Reward Shaping [F] Offline [T] Tree Search [MT] Multi-task

MPE
========================

- `Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments <https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf>`_ **[B][2017]**
- `Learning attentional communication for multi-agent cooperation <https://proceedings.neurips.cc/paper/2018/file/6a8018b3a00b69c008601b8becae392b-Paper.pdf>`_ **[S][2018]**
- `learning when to communicate at scale in multiagent cooperative and competitive tasks <https://arxiv.org/pdf/1812.09755>`_ **[S][2018]**
- `Qtran: Learning to factorize with transformation for cooperative multi-agent reinforcement learning <http://proceedings.mlr.press/v97/son19a/son19a.pdf>`_ **[B][2019]**
- `Robust multi-agent reinforcement learning via minimax deep deterministic policy gradient <https://ojs.aaai.org/index.php/AAAI/article/view/4327/4205>`_ **[R][2019]**
- `Tarmac: Targeted multi-agent communication <http://proceedings.mlr.press/v97/das19a/das19a.pdf>`_ **[S][2019]**
- `Learning Individually Inferred Communication for Multi-Agent Cooperation <https://proceedings.neurips.cc/paper/2020/file/fb2fcd534b0ff3bbed73cc51df620323-Paper.pdf>`_ **[S][2020]**
- `Multi-Agent Game Abstraction via Graph Attention Neural Network <https://ojs.aaai.org/index.php/AAAI/article/view/6211/6067>`_ **[G+S][2020]**
- `Promoting Coordination through Policy Regularization in Multi-Agent Deep Reinforcement Learning <https://proceedings.neurips.cc/paper/2020/file/b628386c9b92481fab68fbf284bd6a64-Paper.pdf>`_ **[E][2020]**
- `Robust Multi-Agent Reinforcement Learning with Model Uncertainty <https://proceedings.neurips.cc/paper/2020/file/774412967f19ea61d448977ad9749078-Paper.pdf>`_ **[R][2020]**
- `Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning <https://proceedings.neurips.cc/paper/2020/file/7967cc8e3ab559e68cc944c44b1cf3e8-Paper.pdf>`_ **[B][2020]**
- `Weighted QMIX Expanding Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning <https://proceedings.neurips.cc/paper/2020/file/73a427badebe0e32caa2e1fc7530b7f3-Paper.pdf>`_ **[B][2020]**
- `Cooperative Exploration for Multi-Agent Deep Reinforcement Learning <http://proceedings.mlr.press/v139/liu21j/liu21j.pdf>`_ **[E][2021]**
- `Multiagent Adversarial Collaborative Learning via Mean-Field Theory <https://ieeexplore.ieee.org/iel7/6221036/9568742/09238422.pdf?casa_token=43-7BP8rsWgAAAAA:ESpZx5Nunchu6Un6vIaVljiJQrSj7tYGWVgx1x3tGvCMkSktx55ZCopEW8VC4SwfjX6RU_KT_c8>`_ **[R][2021]**
- `The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games <https://arxiv.org/pdf/2103.01955?ref=https://githubhelp.com>`_ **[B][2021]**
- `Variational Automatic Curriculum Learning for Sparse-Reward Cooperative Multi-Agent Problems <https://proceedings.neurips.cc/paper/2021/file/503e7dbbd6217b9a591f3322f39b5a6c-Paper.pdf>`_ **[2021]**
- `ToM2C: Target-oriented Multi-agent Communication and Cooperation with Theory of Mind <https://arxiv.org/pdf/2111.09189>`_ **[2021]**
- `Taming Communication and Sample Complexities in Decentralized Policy Evaluation for Cooperative Multi-Agent Reinforcement Learning <https://proceedings.neurips.cc/paper/2021/file/9c51a13764ca629f439f6accbb4ec413-Paper.pdf>`_ **[2021]**
- `SPD: Synergy Pattern Diversifying Oriented Unsupervised Multi-agent Reinforcement Learning <https://openreview.net/pdf?id=jJwy2kcBYv>`_ **[2022]**
- `Distributional Reward Estimation for Effective Multi-Agent Deep Reinforcement Learning <https://openreview.net/pdf?id=4qR780g2Mg>`_ **[2022]**

SMAC
========================

- `Value-Decomposition Networks For Cooperative Multi-Agent Learning <https://arxiv.org/pdf/1706.05296?ref=https://githubhelp.com>`_ **[B][2017]**
- `Counterfactual Multi-Agent Policy Gradients <https://ojs.aaai.org/index.php/AAAI/article/download/11794/11653>`_ **[B][2018]**
- `Multi-Agent Common Knowledge Reinforcement Learning <https://proceedings.neurips.cc/paper/2019/file/f968fdc88852a4a3a27a81fe3f57bfc5-Paper.pdf>`_ **[RG+S][2018]**
- `QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning <http://proceedings.mlr.press/v80/rashid18a/rashid18a.pdf>`_ **[B][2018]**
- `Efficient Communication in Multi-Agent Reinforcement Learning via Variance Based Control <https://proceedings.neurips.cc/paper/2019/file/14cfdb59b5bda1fc245aadae15b1984a-Paper.pdf>`_ **[S][2019]**
- `Exploration with Unreliable Intrinsic Reward in Multi-Agent Reinforcement Learning <https://arxiv.org/pdf/1906.02138>`_ **[P+E][2019]**
- `Learning nearly decomposable value functions via communication minimization <https://arxiv.org/pdf/1910.05366>`_ **[S][2019]**
- `Liir: Learning individual intrinsic reward in multi-agent reinforcement learning <https://proceedings.neurips.cc/paper/2019/file/07a9d3fed4c5ea6b17e80258dee231fa-Paper.pdf>`_ **[P][2019]**
- `MAVEN: Multi-Agent Variational Exploration <https://proceedings.neurips.cc/paper/2019/file/f816dc0acface7498e10496222e9db10-Paper.pdf>`_ **[E][2019]**
- `Adaptive learning A new decentralized reinforcement learning approach for cooperative multiagent systems <https://ieeexplore.ieee.org/iel7/6287639/8948470/09102277.pdf>`_ **[B][2020]**
- `Counterfactual Multi-Agent Reinforcement Learning with Graph Convolution Communication <https://arxiv.org/pdf/2004.00470>`_ **[S+G][2020]**
- `Deep implicit coordination graphs for multi-agent reinforcement learning <https://arxiv.org/pdf/2006.11438>`_ **[G][2020]**
- `DOP: Off-policy multi-agent decomposed policy gradients <https://openreview.net/pdf?id=6FqKiVAdI3Y>`_ **[B][2020]**
- `F2a2: Flexible fully-decentralized approximate actor-critic for cooperative multi-agent reinforcement learning <https://arxiv.org/pdf/2004.11145>`_ **[B][2020]**
- `From few to more Large-scale dynamic multiagent curriculum learning <https://ojs.aaai.org/index.php/AAAI/article/view/6221/6083>`_ **[MT][2020]**
- `Learning structured communication for multi-agent reinforcement learning <https://arxiv.org/pdf/2002.04235>`_ **[S+G][2020]**
- `Learning efficient multi-agent communication: An information bottleneck approach <http://proceedings.mlr.press/v119/wang20i/wang20i.pdf>`_ **[S][2020]**
- `On the robustness of cooperative multi-agent reinforcement learning <https://ieeexplore.ieee.org/iel7/9283745/9283819/09283830.pdf?casa_token=k2lORHebFEUAAAAA:kmTJ2M4Q67hwRz8fh6LhgoXgwZLPy_idCgBmXDxBjzcJBgnYuLmCc7iDS8KTjbVcRPmal-jV9sM>`_ **[R][2020]**
- `Qatten: A general framework for cooperative multiagent reinforcement learning <https://arxiv.org/pdf/2002.03939>`_ **[B][2020]**
- `Revisiting parameter sharing in multi-agent deep reinforcement learning <https://arxiv.org/pdf/2005.13625>`_ **[RG][2020]**
- `Qplex: Duplex dueling multi-agent q-learning <https://arxiv.org/pdf/2008.01062>`_ **[B][2020]**
- `ROMA: Multi-Agent Reinforcement Learning with Emergent Roles <https://arxiv.org/pdf/2003.08039>`_ **[RG][2020]**
- `Towards Understanding Cooperative Multi-Agent Q-Learning with Value Factorization <https://proceedings.neurips.cc/paper/2021/file/f3f1fa1e4348bfbebdeee8c80a04c3b9-Paper.pdf>`_ **[B][2021]**
- `Contrasting centralized and decentralized critics in multi-agent reinforcement learning <https://arxiv.org/pdf/2102.04402>`_ **[B][2021]**
- `Learning in nonzero-sum stochastic games with potentials <http://proceedings.mlr.press/v139/mguni21a/mguni21a.pdf>`_ **[B][2021]**
- `Natural emergence of heterogeneous strategies in artificially intelligent competitive teams <https://arxiv.org/pdf/2007.03102>`_ **[S+G][2021]**
- `Rode: Learning roles to decompose multi-agent tasks <https://arxiv.org/pdf/2010.01523?ref=https://githubhelp.com>`_ **[RG][2021]**
- `SMIX(λ): Enhancing Centralized Value Functions for Cooperative Multiagent Reinforcement Learning <https://ieeexplore.ieee.org/iel7/5962385/6104215/09466372.pdf?casa_token=TdedVHwLvL4AAAAA:kGSnPCM1wQMte1gloaEBUhgD9kUP1FA3mf1TZ931e7W1RqFAr0ewePlhHkEEEArHva6SikWDFA4>`_ **[B][2021]**
- `Tesseract: Tensorised Actors for Multi-Agent Reinforcement Learning <http://proceedings.mlr.press/v139/mahajan21a/mahajan21a.pdf>`_ **[B][2021]**
- `The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games <https://arxiv.org/pdf/2103.01955?ref=https://githubhelp.com>`_ **[B][2021]**
- `UPDeT: Universal Multi-agent Reinforcement Learning via Policy Decoupling with Transformers <https://openreview.net/pdf?id=v9c7hr9ADKx>`_ **[MT][2021]**
- `Randomized Entity-wise Factorization for Multi-Agent Reinforcement Learning <http://proceedings.mlr.press/v139/iqbal21a/iqbal21a.pdf>`_ **[MT][2021]**
- `Cooperative Multi-Agent Transfer Learning with Level-Adaptive Credit Assignment <https://arxiv.org/pdf/2106.00517?ref=https://githubhelp.com>`_ **[MT][2021]**
- `Uneven: Universal value exploration for multi-agent reinforcement learning <http://proceedings.mlr.press/v139/gupta21a/gupta21a.pdf>`_ **[B][2021]**
- `Value-decomposition multi-agent actor-critics <https://www.aaai.org/AAAI21Papers/AAAI-2412.SuJ.pdf>`_ **[B][2021]**
- `RMIX: Learning Risk-Sensitive Policies for Cooperative Reinforcement Learning Agents <https://proceedings.neurips.cc/paper/2021/file/c2626d850c80ea07e7511bbae4c76f4b-Paper.pdf>`_ **[2021]**
- `Regularized Softmax Deep Multi-Agent Q-Learning <https://proceedings.neurips.cc/paper/2021/file/0a113ef6b61820daa5611c870ed8d5ee-Paper.pdf>`_ **[2021]**
- `Policy Regularization via Noisy Advantage Values for Cooperative Multi-agent Actor-Critic methods <https://arxiv.org/pdf/2106.14334>`_ **[2021]**
- `ALMA: Hierarchical Learning for Composite Multi-Agent Tasks <https://arxiv.org/pdf/2205.14205>`_ **[2022]**
- `PAC: Assisted Value Factorisation with Counterfactual Predictions in Multi-Agent Reinforcement Learning <https://arxiv.org/pdf/2206.11420>`_ **[2022]**
- `Rethinking Individual Global Max in Cooperative Multi-Agent Reinforcement Learning <https://arxiv.org/pdf/2209.09640>`_ **[2022]**
- `Surprise Minimizing Multi-Agent Learning with Energy-based Models <https://openreview.net/pdf?id=vkhYWVtfcSQ>`_ **[2022]**
- `Heterogeneous Skill Learning for Multi-agent Tasks <https://openreview.net/pdf?id=5VCT-DptDTs>`_ **[2022]**
- `SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning <https://arxiv.org/pdf/2105.15013>`_ **[2022]**
- `Self-Organized Group for Cooperative Multi-agent Reinforcement Learning <https://openreview.net/pdf?id=hd5KRowT3oB>`_ **[2022]**
- `ResQ: A Residual Q Function-based Approach for Multi-Agent Reinforcement Learning Value Factorization <https://openreview.net/pdf?id=bdnZ_1qHLCW>`_ **[2022]**
- `Efficient Multi-agent Communication via Self-supervised Information Aggregation <https://openreview.net/pdf?id=bdnZ_1qHLCW>`_ **[2022]**
- `Episodic Multi-agent Reinforcement Learning with Curiosity-Driven Exploration <https://proceedings.neurips.cc/paper/2021/file/1e8ca836c962598551882e689265c1c5-Paper.pdf>`_ **[2022]**
- `CTDS: Centralized Teacher with Decentralized Student for Multi-Agent Reinforcement Learning <https://arxiv.org/pdf/2203.08412>`_ **[2022]**


MAMuJoCo
========================

- `FACMAC: Factored Multi-Agent Centralised Policy Gradients <https://arxiv.org/pdf/2003.06709>`_ **[B][2020]**
- `Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning <https://arxiv.org/pdf/2109.11251>`_ **[B][2021]**
- `A Game-Theoretic Approach to Multi-Agent Trust Region Optimization <https://arxiv.org/pdf/2106.06828>`_ **[2021]**
- `Settling the Variance of Multi-Agent Policy Gradients <https://proceedings.neurips.cc/paper/2021/file/6fe6a8a6e6cb710584efc4af0c34ce50-Paper.pdf>`_ **[2021]**
- `Graph-Assisted Predictive State Representations for Multi-Agent Partially Observable Systems <https://openreview.net/pdf?id=PLDOnFoVm4>`_ **[2022]**
- `Order Matters: Agent-by-agent Policy Optimization <https://arxiv.org/pdf/2302.06205>`_ **[2023]**

Google Research Football
========================


- `Adaptive Inner-reward Shaping in Sparse Reward Games <https://ieeexplore.ieee.org/iel7/9200848/9206590/09207302.pdf?casa_token=T6Xp9_s07OwAAAAA:ECy-wfIOoMq60Mkk3qfitWlSzslNTC5mBkHtVLu1SmJ9STDErl7OYjoptRKU6PMsqh7_4cbP6Jk>`_ **[P][2020]**
- `Factored action spaces in deep reinforcement learning <https://openreview.net/pdf?id=naSAkn2Xo46>`_ **[B][2021]**
- `Semantic Tracklets An Object-Centric Representation for Visual Multi-Agent Reinforcement Learning <https://ieeexplore.ieee.org/iel7/9635848/9635849/09636592.pdf?casa_token=x8RsQf74KUUAAAAA:lp6vsCBIaMlYbhP4xoIM2279USMn3-KW73DxyhejGOz-hiG2kDRqQIrNSABy6IlAYdU4BvRqAnc>`_ **[B][2021]**
- `TiKick: Towards Playing Multi-agent Football Full Games from Single-agent Demonstrations <https://arxiv.org/pdf/2110.04507>`_ **[F][2021]**
- `Celebrating Diversity in Shared Multi-Agent Reinforcement Learning <https://proceedings.neurips.cc/paper/2021/file/20aee3a5f4643755a79ee5f6a73050ac-Paper.pdf>`_ **[2021]**
- `Mingling Foresight with Imagination: Model-Based Cooperative Multi-Agent Reinforcement Learning <https://arxiv.org/pdf/2204.09418>`_ **[2022]**

Pommerman
========================

- `Using Monte Carlo Tree Search as a Demonstrator within Asynchronous Deep RL <https://arxiv.org/pdf/1812.00045>`_ **[I+T][2018]**
- `Accelerating Training in Pommerman with Imitation and Reinforcement Learning <https://arxiv.org/pdf/1911.04947>`_ **[I][2019]**
- `Agent Modeling as Auxiliary Task for Deep Reinforcement Learning <https://ojs.aaai.org/index.php/AIIDE/article/download/5221/5077/>`_ **[S][2019]**
- `Backplay: man muss immer umkehren <https://arxiv.org/pdf/1807.06919.pdf%20http://arxiv.org/abs/1807.06919>`_ **[I][2019]**
- `Terminal Prediction as an Auxiliary Task for Deep Reinforcement Learning <https://ojs.aaai.org/index.php/AIIDE/article/download/5222/5078>`_ **[B][2019]**
- `Adversarial Soft Advantage Fitting Imitation Learning without Policy Optimization <https://proceedings.neurips.cc/paper/2020/file/9161ab7a1b61012c4c303f10b4c16b2c-Paper.pdf>`_ **[B][2020]**
- `Evolutionary Reinforcement Learning for Sample-Efficient Multiagent Coordination <http://proceedings.mlr.press/v119/majumdar20a/majumdar20a.pdf>`_ **[B][2020]**

LBF & RWARE
========================


- `Shared Experience Actor-Critic for Multi-Agent Reinforcement Learning <https://proceedings.neurips.cc/paper/2020/file/7967cc8e3ab559e68cc944c44b1cf3e8-Paper.pdf>`_ **[B][2020]**
- `Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks <https://arxiv.org/pdf/2006.07869>`_ **[B][2021]**
- `Learning Altruistic Behaviors in Reinforcement Learning without External Rewards <https://arxiv.org/pdf/2107.09598>`_ **[B][2021]**
- `Scaling Multi-Agent Reinforcement Learning with Selective Parameter Sharing <http://proceedings.mlr.press/v139/christianos21a/christianos21a.pdf>`_ **[RG][2021]**
- `LIGS: Learnable Intrinsic-Reward Generation Selection for Multi-Agent Learning <https://arxiv.org/pdf/2112.02618>`_ **[2022]**

MetaDrive
========================

- `Learning to Simulate Self-Driven Particles System with Coordinated Policy Optimization <https://proceedings.neurips.cc/paper/2021/file/594ca7adb3277c51a998252e2d4c906e-Paper.pdf>`_ **[B][2021]**
- `Safe Driving via Expert Guided Policy Optimization <https://proceedings.mlr.press/v164/peng22a/peng22a.pdf>`_ **[I][2021]**

Hanabi
========================

- `Bayesian Action Decoder for Deep Multi-Agent Reinforcement Learning <http://proceedings.mlr.press/v97/foerster19a/foerster19a.pdf>`_ **[B][2019]**
- `Re-determinizing MCTS in Hanabi <https://ieeexplore.ieee.org/iel7/8844551/8847948/08848097.pdf?casa_token=nZ3ZAeyS1-kAAAAA:3FBwAb2lMlQ_ClJIlycoVsensDQFE0pqMeQ8PvMc15Bzoam9inGlWBJmT6D9bKjF1WUL7k5IkS0>`_ **[S+T][2019]**
- `Diverse Agents for Ad-Hoc Cooperation in Hanabi <https://ieeexplore.ieee.org/iel7/8844551/8847948/08847944.pdf?casa_token=oDFhRxwd0XIAAAAA:Vq6oBEA6fotbST9N-RkThJjY5URVVvnwQ8Y0mt1JiD9uLXmXMxt7k8Dqt-VghWJzK8fOgdXFbH0>`_ **[B][2019]**
- `Joint Policy Search for Multi-agent Collaboration with Imperfect Information <https://proceedings.neurips.cc/paper/2020/file/e64f346817ce0c93d7166546ac8ce683-Paper.pdf>`_ **[T][20209]**
- `Off-Belief Learning <http://proceedings.mlr.press/v139/hu21c/hu21c.pdf>`_ **[B][2021]**
- `The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games <https://arxiv.org/pdf/2103.01955?ref=https://githubhelp.com>`_ **[B][2021]**
- `2021 Trajectory Diversity for Zero-Shot Coordination <http://proceedings.mlr.press/v139/lupu21a/lupu21a.pdf>`_ **[B][2021]**

MAgent
========================

- `Mean field multi-agent reinforcement learning <http://proceedings.mlr.press/v80/yang18d/yang18d.pdf>`_ **[B][2018]**
- `Graph convolutional reinforcement learning <https://arxiv.org/pdf/1810.09202>`_ **[B][2018]**
- `Factorized q-learning for large-scale multi-agent systems <https://dl.acm.org/doi/pdf/10.1145/3356464.3357707?casa_token=AQbNTCy_0KcAAAAA:iRFZ9HPbGUw-nqo9g--rsQoripkpVU8CPuiIC4n_1ffrmyYm1jwvfNRc_tygCqbFNLPSc131yojiWw>`_ **[B][2019]**
- `From few to more Large-scale dynamic multiagent curriculum learning <https://ojs.aaai.org/index.php/AAAI/article/view/6221/6083>`_ **[MT][2020]**

Other Tasks
========================

- `Learning Fair Policies in Decentralized Cooperative Multi-Agent Reinforcement Learning <http://proceedings.mlr.press/v139/zimmer21a/zimmer21a.pdf>`_ **[2020]**
- `Contrasting Centralized and Decentralized Critics in Multi-Agent Reinforcement Learning <https://arxiv.org/pdf/2102.04402>`_ **[2021]**
- `Learning to Ground Multi-Agent Communication with Autoencoders <https://proceedings.neurips.cc/paper/2021/file/80fee67c8a4c4989bf8a580b4bbb0cd2-Paper.pdf>`_ **[2021]**
- `Latent Variable Sequential Set Transformers For Joint Multi-Agent Motion Prediction <https://arxiv.org/pdf/2104.00563>`_ **[2021]**
- `Learning to Share in Multi-Agent Reinforcement Learning <https://arxiv.org/pdf/2112.08702>`_ **[2021]**
- `Resilient Multi-Agent Reinforcement Learning with Adversarial Value Decomposition <https://ojs.aaai.org/index.php/AAAI/article/download/17348/17155>`_ **[2021]**
- `Multi-Agent MDP Homomorphic Networks <https://arxiv.org/pdf/2110.04495>`_ **[2021]**
- `Multi-Agent Reinforcement Learning for Active Voltage Control on Power Distribution Networks <https://proceedings.neurips.cc/paper/2021/file/1a6727711b84fd1efbb87fc565199d13-Paper.pdf>`_ **[2021]**
- `Multi-Agent Reinforcement Learning in Stochastic Networked Systems <https://proceedings.neurips.cc/paper/2021/file/412604be30f701b1b1e3124c252065e6-Paper.pdf>`_ **[2021]**
- `Mirror Learning: A Unifying Framework of Policy Optimisation <https://proceedings.mlr.press/v162/grudzien22a/grudzien22a.pdf>`_ **[2022]**
- `E-MAPP: Efficient Multi-Agent Reinforcement Learning with Parallel Program Guidance <https://arxiv.org/pdf/2212.02064>`_ **[2022]**
- `Shield Decentralization for Safe Multi-Agent Reinforcement Learning <https://par.nsf.gov/servlets/purl/10389823>`_ **[2022]**
- `Provably Efficient Offline Multi-agent Reinforcement Learning via Strategy-wise Bonus <https://arxiv.org/pdf/2206.00159>`_ **[2022]**
- `Asynchronous Actor-Critic for Multi-Agent Reinforcement Learning <https://arxiv.org/pdf/2209.10113>`_ **[2022]**
- `Near-Optimal Multi-Agent Learning for Safe Coverage Control <https://arxiv.org/pdf/2210.06380>`_ **[2022]**
- `Multi-agent Dynamic Algorithm Configuration <https://arxiv.org/pdf/2210.06835>`_ **[2022]**

New Environments
========================

- `SMACv2: A New Benchmark for Cooperative Multi-Agent Reinforcement Learning <https://arxiv.org/pdf/2212.07489.pdf>`_ **[2022]**
- `MATE: Benchmarking Multi-Agent Reinforcement Learning in Distributed Target Coverage Control <https://openreview.net/pdf?id=SyoUVEyzJbE>`_ **[2022]**
- `Nocturne: a scalable driving benchmark for bringing multi-agent learning one step closer to the real world <https://arxiv.org/pdf/2206.09889>`_ **[2022]**
- `GoBigger: A Scalable Platform for Cooperative-Competitive Multi-Agent Interactive Simulation <https://openreview.net/pdf?id=NnOZT_CR26Z>`_ **[2023]**