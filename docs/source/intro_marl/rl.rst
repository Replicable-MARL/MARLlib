.. _part1:

***************************************
Part 1. Single Agent (Deep) RL
***************************************

There have been many great RL tutorials and open-sourced repos where you can find both the principle of different RL algorithms and
the implementation details. In this part, we are going to have a quick review of what single agent RL has done. If you are quite familiar with RL,
especially model-free RL, you can skip this part and go to :ref:`part2`.

.. contents::
    :local:
    :depth: 2

Standard Reinforcement Learning
===================================

Reinforcement Learning is focused on goal-directed learning from interaction.
The learning entity must discover for itself which strategy produce the greatest reward by "trial and error".

**Key Concepts**:

- **Agent** represents the solution, making decisions (actions) to solve decision-making problems under uncertainty.
- **Environment** is the representation of a problem which responds with the consequences of agent decisions.
- **State** is a set of variables which fully describe the environment.
- **Observation** is part of the state. Commonly, agent doesn't have access to the full state of the environment.
- **Action** is made by the agent, influencing the environment state.
- **Transition Function** is the mapping responsible for action-state chang .
- **Reward** is a signal provided by the environment as a direct evaluation to the agent's actions.
- **Episode** only exists when a task have a natural ending. A sequence of **timesteps** from the beginning to the end of the task forms a task episode.


Deep Reinforcement Learning(DRL)
================================

Deep Reinforcement Learning (DRL) is the combination of Reinforcement Learning and Deep Learning.
It can solve a wide range of complex decision-making tasks that were previously out of reach for a machine to solve real-world problems with human-like intelligence.

Deep Learning(DL)
---------------------

Deep learning can learn from a training set and then applying that learning to a new data set.
Deep learning is well known for its function fitting ability which can infinitely approximated to the optimal mapping function for a high dimensional problem.


DL + RL
---------------------------

Deep neural networks enables RL with state representation and/or function approximation for value function, policy, and so on.
Deep reinforcement learning incorporates deep learning into the solution, allowing agents to make decisions from unstructured input data without manual engineering of the state space.
A instance of combining Q learning with Deep learning can be found in :ref:`DQN`.


Learning Cycle
-----------------

on-policy algorithm: data collection - form a batch - policy optimization - data collection

off-policy algorithm: data collection - replay buffer - sample a batch - policy optimization - data collection

- **data collection**: agent send an action to environment, environment return some results including observation, state, reward, etc.
- **form a batch**: policy optimization need a batch of data from **data collection** to do stochastic gradient descent (SGD).
- **replay buffer**: data from **data collection** is sent to replay buffer for future optimization use.
- **sample a batch**: sample a batch from **replay buffer** follow some rules.
- **policy optimization**: use the data batch to optimize the policy.

RL/DRL Algorithms
----------------------------

A comprehensive collection of RL algorithms from very old to very new: `Awesome DRL <https://github.com/tigerneil/awesome-deep-rl>`_.

Resources
=================

A great`RL resource guide <https://github.com/aikorea/awesome-rl>`_ including all kinds of RL related surveys, books, open-sourced repos, etc.




