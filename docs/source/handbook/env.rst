.. _env:


*********************************************
Environments
*********************************************

Environment list of MARLlib, including installation and description.

.. contents::
    :local:
    :depth: 1



**Note**: make sure you have read and completed the :ref:`basic-installation` part.


.. _SMAC:


SMAC
==============

.. figure:: ../images/env_smac.png
    :align: center

StarCraft Multi-Agent Challenge (SMAC) is a multi-agent environment for collaborative multi-agent reinforcement learning (MARL) research based on Blizzard's StarCraft II RTS game.
It focuses on decentralized micromanagement scenarios, where an individual RL agent controls each game unit.

Official Link: https://github.com/oxwhirl/smac

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``Original Learning Mode``
     - Cooperative
   * - ``MARLlib Learning Mode``
     - Cooperative + Collaborative
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
   * - ``Reward``
     - Dense / Sparse
   * - ``Agent-Env Interact Mode``
     - Simultaneous

Installation
-----------------

.. code-block:: shell

    bash install_sc2.sh # https://github.com/oxwhirl/pymarl/blob/master/install_sc2.sh
    pip3 install numpy scipy pyyaml matplotlib
    pip3 install imageio
    pip3 install tensorboard-logger
    pip3 install pygame
    pip3 install jsonpickle==0.9.6
    pip3 install setuptools
    pip3 install sacred

    git clone https://github.com/oxwhirl/smac.git
    cd smac
    pip install .

**Note**: the location of the StarcraftII game directory should be pre-defined,
or you can just follow the error log (when the process can not found the game's location)
and put it in the right place.

.. _MAMuJoCo:

MAMuJoCo
==============

.. figure:: ../images/env_mamujoco.png
    :align: center

Multi-Agent Mujoco (MAMuJoCo) is an environment for continuous cooperative multi-agent robotic control.
Based on the popular single-agent robotic MuJoCo control suite provides a wide variety of novel scenarios in which multiple agents within a single robot have to solve a task cooperatively.

Official Link: https://github.com/schroederdewitt/multiagent_mujoco

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``Original Learning Mode``
     - Cooperative
   * - ``MARLlib Learning Mode``
     - Cooperative + Collaborative
   * - ``Observability``
     - Partial
   * - ``Action Space``
     - Continuous
   * - ``Observation Space Dim``
     - 1D
   * - ``Action Mask``
     - No
   * - ``Global State``
     - Yes
   * - ``Global State Space Dim``
     - 1D
   * - ``Reward``
     - Dense
   * - ``Agent-Env Interact Mode``
     - Simultaneous


Installation
-----------------

.. code-block:: shell

    mkdir /home/YourUserName/.mujoco
    cd /home/YourUserName/.mujoco
    wget https://roboti.us/download/mujoco200_linux.zip
    unzip mujoco200_linux.zip
    export LD_LIBRARY_PATH=/home/YourUserName/.mujoco/mujoco200/bin;
    pip install mujoco-py==2.0.2.8

    git clone https://github.com/schroederdewitt/multiagent_mujoco
    cd multiagent_mujoco
    mv multiagent_mujoco /home/YourPathTo/MARLlib/multiagent_mujoco

    # optional
    sudo apt-get install libosmesa6-dev # If you meet GCC error with exit status 1
    pip install patchelf-wrapper

**Note**: To access the MuJoCo API, you may get a mjkey (free now) and put it under /home/YourUserName/.mujoco.






.. _Football:

Google Research Football
================================


.. figure:: ../images/env_football.png
    :align: center


Google Research Football (GRF) is a reinforcement learning environment where agents are trained to play football in an advanced,
physics-based 3D simulator. It also provides support for multiplayer and multi-agent experiments.

Official Link: https://github.com/google-research/football

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``Original Learning Mode``
     - Collaborative + Competitive
   * - ``MARLlib Learning Mode``
     - Cooperative + Collaborative
   * - ``Observability``
     - Full
   * - ``Action Space``
     - Discrete
   * - ``Observation Space Dim``
     - 2D
   * - ``Action Mask``
     - No
   * - ``Global State``
     - No
   * - ``Global State Space Dim``
     - /
   * - ``Reward``
     - Sparse
   * - ``Agent-Env Interact Mode``
     - Simultaneous




Installation
-----------------

Google Research Football is somehow a bit tricky for installation. We wish you good luck.

.. code-block:: shell

    sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip
    python3 -m pip install --upgrade pip setuptools psutil wheel

We provide solutions (may work) for potential bugs

* `Compiler error on /usr/lib/x86_64-linux-gnu/libGL.so <https://github.com/RobotLocomotion/drake/issues/2087>`_
* `apt-get, unmet dependencies, ... "but it is not going to be installed" <https://askubuntu.com/questions/564282/apt-get-unmet-dependencies-but-it-is-not-going-to-be-installed>`_
* `Errors related to Could NOT find Boost <https://github.com/google-research/football/issues/317>`_

.. _MPE:

MPE
==============

.. figure:: ../images/env_mpe.png
    :align: center

Multi-particle Environments (MPE) are a set of communication-oriented environments where particle agents can (sometimes) move,
communicate, see each other, push each other around, and interact with fixed landmarks.

Official Link: https://github.com/openai/multiagent-particle-envs

Our version: https://github.com/Farama-Foundation/PettingZoo/tree/master/pettingzoo/mpe

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``Original Learning Mode``
     - Collaborative + Competitive
   * - ``MARLlib Learning Mode``
     - Cooperative + Collaborative + Competitive + Mixed
   * - ``Observability``
     - Full
   * - ``Action Space``
     - Discrete + Continuous
   * - ``Observation Space Dim``
     - 1D
   * - ``Action Mask``
     - No
   * - ``Global State``
     - No
   * - ``Global State Space Dim``
     - /
   * - ``Reward``
     - Dense
   * - ``Agent-Env Interact Mode``
     - Simultaneous / Asynchronous




Installation
-----------------

We use the pettingzoo version of MPE

.. code-block:: shell

    pip install pettingzoo[mpe]

.. _LBF:

LBF
==============

.. figure:: ../images/env_lbf.png
    :align: center

Level-based Foraging (LBF) is a mixed cooperative-competitive game that focuses on coordinating the agents involved.
Agents navigate a grid world and collect food by cooperating with other agents if needed.

Official Link: https://github.com/semitable/lb-foraging

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``Original Learning Mode``
     - Cooperative + Collaborative
   * - ``MARLlib Learning Mode``
     - Cooperative + Collaborative
   * - ``Observability``
     - Partial
   * - ``Action Space``
     - Discrete
   * - ``Observation Space Dim``
     - 1D
   * - ``Action Mask``
     - No
   * - ``Global State``
     - No
   * - ``Global State Space Dim``
     - /
   * - ``Reward``
     - Dense
   * - ``Agent-Env Interact Mode``
     - Simultaneous

Installation
-----------------

.. code-block:: shell

    pip install lbforaging==1.0.15

.. _RWARE:


RWARE
==============

.. figure:: ../images/env_rware.png
    :align: center

Robot Warehouse (RWARE) simulates a warehouse with robots moving and delivering requested goods.
Real-world applications inspire the simulator, in which robots pick up shelves and deliver them to a workstation.

Official Link: https://github.com/semitable/robotic-warehouse

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``Original Learning Mode``
     - Cooperative
   * - ``MARLlib Learning Mode``
     - Cooperative + Collaborative
   * - ``Observability``
     - Partial
   * - ``Action Space``
     - Discrete
   * - ``Observation Space Dim``
     - 1D
   * - ``Action Mask``
     - No
   * - ``Global State``
     - No
   * - ``Global State Space Dim``
     - /
   * - ``Reward``
     - Sparse
   * - ``Agent-Env Interact Mode``
     - Simultaneous

Installation
-----------------

.. code-block:: shell

    pip install rware==1.0.1

.. _MAgent:


MAgent
==============

.. figure:: ../images/env_magent.png
    :align: center

MAgent is a set of environments where large numbers of pixel agents in a grid world interact in battles or other competitive scenarios.

Official Link: https://www.pettingzoo.ml/magent

Our version: https://github.com/Farama-Foundation/PettingZoo/tree/master/pettingzoo/mpe

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``Original Learning Mode``
     - Collaborative + Competitive
   * - ``MARLlib Learning Mode``
     - Collaborative + Competitive
   * - ``Observability``
     - Partial
   * - ``Action Space``
     - Discrete
   * - ``Observation Space Dim``
     - 2D
   * - ``Action Mask``
     - No
   * - ``Global State``
     - MiniMap
   * - ``Global State Space Dim``
     - 2D
   * - ``Reward``
     - Dense
   * - ``Agent-Env Interact Mode``
     - Simultaneous / Asynchronous

Installation
-----------------

.. code-block:: shell

    pip install pettingzoo[magent]

.. _Pommerman:



Pommerman
==============

.. figure:: ../images/env_pommerman.png
    :align: center

Pommerman is stylistically similar to Bomberman, the famous game from Nintendo.
Pommerman's FFA is a simple but challenging setup for engaging adversarial research where coalitions are possible,
and Team asks agents to be able to work with others to accomplish a shared but competitive goal.

Official Link: https://github.com/MultiAgentLearning/playground

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``Original Learning Mode``
     - Collaborative + Competitive
   * - ``MARLlib Learning Mode``
     - Cooperative + Collaborative + Competitive + Mixed
   * - ``Observability``
     - Full
   * - ``Action Space``
     - Discrete
   * - ``Observation Space Dim``
     - 2D
   * - ``Action Mask``
     - No
   * - ``Global State``
     - No
   * - ``Global State Space Dim``
     - /
   * - ``Reward``
     - Sparse
   * - ``Agent-Env Interact Mode``
     - Simultaneous

Installation
-----------------

.. code-block:: shell

    git clone https://github.com/MultiAgentLearning/playground
    cd playground
    pip install .
    cd /home/YourPathTo/MARLlib/patch
    python add_patch.py --pommerman
    pip install gym==0.21.0

.. _MetaDrive:



MetaDrive
==============

.. figure:: ../images/env_metadrive.png
    :align: center

MetaDrive is a driving simulator that supports generating infinite scenes with various road maps and
traffic settings to research generalizable RL. It provides accurate physics simulation and multiple sensory inputs,
including Lidar, RGB images, top-down semantic maps, and first-person view images.

Official Link: https://github.com/decisionforce/metadrive

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``Original Learning Mode``
     - Collaborative
   * - ``MARLlib Learning Mode``
     - Collaborative
   * - ``Observability``
     - Partial
   * - ``Action Space``
     - Continuous
   * - ``Observation Space Dim``
     - 1D
   * - ``Action Mask``
     - No
   * - ``Global State``
     - No
   * - ``Global State Space Dim``
     - /
   * - ``Reward``
     - Dense
   * - ``Agent-Env Interact Mode``
     - Simultaneous


Installation
-----------------

.. code-block:: shell

    pip install metadrive-simulator==0.2.3

.. _Hanabi:

Hanabi
==============

.. figure:: ../images/env_hanabi.png
    :align: center

Hanabi is a cooperative card game created by French game designer Antoine Bauza.
Players are aware of other players' cards but not their own and attempt to play a series of cards in a
specific order to set off a simulated fireworks show.

Official Link: https://github.com/deepmind/hanabi-learning-environment

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``Original Learning Mode``
     - Collaborative
   * - ``MARLlib Learning Mode``
     - Collaborative
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
   * - ``Reward``
     - Dense
   * - ``Agent-Env Interact Mode``
     - Asynchronous

Installation
-----------------

From `MAPPO official site <https://github.com/marlbenchmark/on-policy>`_

The environment code for Hanabi is developed from the open-source environment code but has been slightly modified to fit the algorithms used here.
To install, execute the following:

.. code-block:: shell

    pip install cffi
    cd /home/YourPathTo/MARLlib/patch/hanabi
    mkdir build
    cd build
    cmake ..
    make -j


.. _MATE:

MATE
==============

.. figure:: ../images/env_mate.png
    :align: center

Multi-Agent Tracking Environment (MATE) is an asymmetric two-team zero-sum stochastic game with partial observations, and each team has multiple agents (multiplayer). Intra-team communications are allowed, but inter-team communications are prohibited. It is cooperative among teammates, but it is competitive among teams (opponents).

Official Link: https://github.com/XuehaiPan/mate

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``Original Learning Mode``
     - Cooperative + Mixed
   * - ``MARLlib Learning Mode``
     - Cooperative + Mixed
   * - ``Observability``
     - Partial
   * - ``Action Space``
     - Discrete + Continuous
   * - ``Observation Space Dim``
     - 1D
   * - ``Action Mask``
     - No
   * - ``Global State``
     - No
   * - ``Global State Space Dim``
     - /
   * - ``Reward``
     - Dense
   * - ``Agent-Env Interact Mode``
     - Simultaneous


Installation
-----------------

.. code-block:: shell

    pip3 install git+https://github.com/XuehaiPan/mate.git#egg=mate


.. _GoBigger:

GoBigger
==============
.. only:: html

    .. figure:: images/env_gobigger.gif
       :width: 320
       :align: center


GoBigger is a game engine that offers an efficient and easy-to-use platform for agar-like game development. It provides a variety of interfaces specifically designed for game AI development. The game mechanics of GoBigger are similar to those of Agar, a popular massive multiplayer online action game developed by Matheus Valadares of Brazil. The objective of GoBigger is for players to navigate one or more circular balls across a map, consuming Food Balls and smaller balls to increase their size while avoiding larger balls that can consume them. Each player starts with a single ball, but can divide it into two when it reaches a certain size, giving them control over multiple balls.
Official Link: https://github.com/opendilab/GoBigger

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``Original Learning Mode``
     - Cooperative + Mixed
   * - ``MARLlib Learning Mode``
     - Cooperative + Mixed
   * - ``Observability``
     - Partial + Full
   * - ``Action Space``
     - Continuous
   * - ``Observation Space Dim``
     - 1D
   * - ``Action Mask``
     - No
   * - ``Global State``
     - No
   * - ``Global State Space Dim``
     - /
   * - ``Reward``
     - Dense
   * - ``Agent-Env Interact Mode``
     - Simultaneous


Installation
-----------------

.. code-block:: shell

    conda install -c opendilab gobigger