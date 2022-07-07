.. _algorithms:

Algorithms
===================

We provide a comprehensive view of algorithms how we collected and implemented MARL algorithms based on RLlib.

An algorithm may not be available for every different environments. You can check it here to get detailed information.

Independent Learning
----------------------

Extending standard single agent RL algorithm to multi-agent setting is a natural idea, except that the condition of Markov Decision Process (MDP) is not satisfied any more.
As RLlib has provided a great number of single agent RL algorithms under its highly modularized framework, we directly adopt them in our framework.

The algorithms that are from RLlib implementation:

* Policy Gradient (PG)
* Advanced Actor Critic (A2C)
* Proximal Policy Optimization (PPO)

The algorithms that we extended for RLlib:

* Deep Deterministic Policy Gradients (DDPG)

    * wrap it with RNN support

* Trust Region Policy Optimization (TRPO)

    * adopt from `here <https://github.com/0xangelo/raylab/tree/master/raylab/agents/trpo>`_ with extension

* Independent Q Learning (IQL)

    * change the execution plan and replay buffer to let it align with `pymarl <https://github.com/oxwhirl/pymarl>`_

All other algorithms are built based on these six independent learning algorithms.

Centralized Critic
----------------------

We strongly recommend using `conda <https://docs.conda.io/en/latest/miniconda.html>`_ to manage your dependencies, and avoid version conflicts. Here we show the example of building python 3.7 based conda environment.

.. code-block:: shell

    conda create -n marllib python==3.7 -y
    conda activate marllib

    # install dependencies
    cmake --version # must be >=3.12
    clang++ --version   # must be >=7.0.0
    sudo apt-get install graphviz cmake clang

    # install marllib
    pip install -e .

Value Decomposition (Joint Q/Critic Learning)
-----------------------------------------------

We strongly recommend using `conda <https://docs.conda.io/en/latest/miniconda.html>`_ to manage your dependencies, and avoid version conflicts. Here we show the example of building python 3.7 based conda environment.

.. code-block:: shell

    conda create -n marllib python==3.7 -y
    conda activate marllib

    # install dependencies
    cmake --version # must be >=3.12
    clang++ --version   # must be >=7.0.0
    sudo apt-get install graphviz cmake clang

    # install marllib
    pip install -e .

Google Research Football
-----------------

We strongly recommend using `conda <https://docs.conda.io/en/latest/miniconda.html>`_ to manage your dependencies, and avoid version conflicts. Here we show the example of building python 3.7 based conda environment.

.. code-block:: shell

    conda create -n marllib python==3.7 -y
    conda activate marllib

    # install dependencies
    cmake --version # must be >=3.12
    clang++ --version   # must be >=7.0.0
    sudo apt-get install graphviz cmake clang

    # install marllib
    pip install -e .

External Environments
---------------------

External environments are integrated in MARLlib, such as `StarCraftII <https://github.com/oxwhirl/smac>`_ and `Mujoco <https://mujoco.org/>`_. You can intall them by following the official guides on their project homepage.


Development requirements
------------------------

For users who wanna contribute to our repository, run ``pip install -e .[dev]`` to complete the development dependencies, also refer the contributing guide.

