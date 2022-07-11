.. _environments:

Env Description
=======================

Brief introduction of 10 different environments in MARLLib.

SMAC
-----------------

StarCraft Multi-Agent Challenge (SMAC) is a multi-agent environment for research in the field of collaborative multi-agent reinforcement learning (MARL) based on Blizzard's StarCraft II RTS game.
It concentrates on decentralized micromanagement scenarios, where an individual RL agent controls each game unit.

Official Link: https://github.com/oxwhirl/smac

.. list-table:: SMAC Environment Features
   :widths: 25 25
   :header-rows: 0

   * - ``Learning Mode``
     - Cooperative
   * - ``Observability``
     - Partial
   * - ``Action Space``
     - Discrete
   * - ``Observation Space Dim``
     - 1D
   * - ``Action Mask``
     - Yes
   * - ``Global State``
     - Yes
   * - ``Global State Space Dim``
     - 1D



MaMujoco
-----------------

Multi-Agent Mujoco (MaMujoco) is an environment for continuous cooperative multi-agent robotic control.
Based on the popular single-agent robotic MuJoCo control suite provides a wide variety of novel scenarios in which multiple agents within a single robot have to solve a task cooperatively.

Official Link: https://github.com/schroederdewitt/multiagent_mujoco

Google Research Football
-----------------------------

Google Research Football (GRF) is a reinforcement learning environment where agents are trained to play football in an advanced,
physics-based 3D simulator. It also provides support for multiplayer and multi-agent experiments.

Official Link: https://github.com/google-research/football

MPE
-----------------

Multi-particle Environments (MPE) are a set of communication-oriented environments where particle agents can (sometimes) move,
communicate, and see each other, push each other around, and interact with fixed landmarks.

Official Link: https://github.com/openai/multiagent-particle-envs

Our version: https://github.com/Farama-Foundation/PettingZoo/tree/master/pettingzoo/mpe

LBF
---------------------

Level-based Foraging (LBF) is a mixed cooperative-competitive game that focuses on the coordination of the agents involved.
Agents navigate a grid world and collect food by cooperating with other agents if needed.

Official Link: https://github.com/semitable/lb-foraging

RWARE
------------------------

Robot Warehouse (RWARE) simulates a warehouse with robots moving and delivering requested goods.
Real-world applications inspire the simulator, in which robots pick up shelves and deliver them to a workstation.

Official Link: https://github.com/semitable/robotic-warehouse

MAgent
------------------------

MAgent is a set of environments where large numbers of pixel agents in a grid world interact in battles or other competitive scenarios.

Official Link: https://www.pettingzoo.ml/magent

Our version: https://github.com/Farama-Foundation/PettingZoo/tree/master/pettingzoo/mpe

Pommerman
------------------------

Pommerman \cite{pommerman}} is stylistically similar to Bomberman, the famous game from Nintendo.
Pommerman's FFA is a simple but challenging setup for engaging adversarial research where coalitions are possible,
and Team asks agents to be able to work with others to accomplish a shared but competitive goal.

Official Link: https://github.com/MultiAgentLearning/playground

MetaDrive
------------------------

MetaDrive is a driving simulator that supports generating infinite scenes with various road maps and
traffic settings for the research of generalizable RL. It provides accurate physics simulation and multiple sensory inputs,
including Lidar, RGB images, top-down semantic maps, and first-person view images.

Official Link: https://github.com/decisionforce/metadrive

Hanabi
------------------------

Hanabi is a cooperative card game created by French game designer Antoine Bauza.
Players are aware of other players' cards but not their own and attempt to play a series of cards in a
specific order to set off a simulated fireworks show.

Official Link: https://github.com/deepmind/hanabi-learning-environment