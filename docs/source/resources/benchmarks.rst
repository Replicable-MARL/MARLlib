.. _concept:

***************************************
Existing Benchmarks
***************************************

We collect most of the existing MARL benchmarks here with brief introduction.

.. contents::
    :local:
    :depth: 3

[B] Basic [S] Information Sharing [RG] Behavior/Role Grouping [I] Imitation [G] Graph [E] Exploration [R] Robust [P] Reward Shaping [F] Offline [T] Tree Search [MT] Multi-task

PyMARL
========================

Github: https://github.com/oxwhirl/pymarl

PyMARL is the first and the most well-known benchmark for micro-management in Starcraft II.
It contains five available MARL algorithms including IQL, VDN, QMIX, QTRAN, and COMA.
 The code is easy to read and understand. However, it only supports SMAC, which is a cooperative MAS with discrete action space.


PyMARL2
========================

Github: https://github.com/hijkzzz/pymarl2

PyMARL2 extends the original PyMARL to more available algorithms, including 15 code-level tricks in MARL training,
7 value-based methods, and 5 gradient-based methods. It provides a more comprehensive overview of existing algorithms
on SMAC and gives a study on how to finetune the MARL algorithms for better performance.


MARL-Algorithms
========================

Github: https://github.com/starry-sky6688/MARL-Algorithms

MARL-Algorithm is another famous MARL benchmark focusing on SMAC,
it includes 9 different MARL algorithms covering broader research topics including communication-based, graph-based and multi-task training in MARL.

EPyMARL
========================

Github: https://github.com/uoe-agents/epymarl

EPyMARL is the first cooperative MARL benchmark that contains more than one task. EPyMARL is also built upon PyMARL,
it contains three more multi-agent tasks including Multi-particle Environment (MPE), Level-based Foraging (LBF), and Robot Warehouse (RWARE) other than SMAC.
All four tasks are cooperative tasks and take discrete action to interact with the environment. EPyMARL is also the first to categorize the existing MARL algorithms into three types:
Independent Learning, Centralized Critic, and Value Decomposition.
EPyMARL successfully unifies the current cooperative MARL tasks and provides a great benchmark for people to understand and compare the cooperative MARL algorithms.

Marlbenchmark
========================

Github: https://github.com/marlbenchmark/on-policy

Marlbenchmark is the first MARL benchmark that contains mixed MARL tasks including cooperative (SMAC, Hanabi), collaborative (MPE), and competitive (MPE) with both on-policy and off-policy algorithms including VDN, QMIX, MADDPG, and MATD3.
It is also the first implementation of MAPPO, which combine the state-of-the-art policy-gradient RL method PPO and centralized critic framework, and performs surprisingly well in practice.

MAlib
========================

Github: https://github.com/sjtu-marl/malib

MALib is a parallel framework of population-based learning nested with (multi-agent) reinforcement learning (RL) methods, such as Policy Space Response Oracle,
Self-Play and Neural Fictitious Self-Play. MALib provides higher-level abstractions of MARL training paradigms, which enables efficient code reuse and flexible deployments on different distributed computing paradigms.
The design of MALib also strives to promote the research of other multi-agent learning, including multi-agent imitation learning and model-based MARL.

MARLlib
========================

Github: https://github.com/Replicable-MARL/MARLlib

Please refer to :ref:`intro`.



