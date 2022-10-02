.. _intro:

Introduction
============

**Multi-Agent RLlib (MARLlib)** is **a comprehensive Multi-Agent Reinforcement Learning algorithm library** based on `Ray <https://github.com/ray-project/ray>`_ and one of its toolkits `RLlib <https://github.com/ray-project/ray/tree/master/rllib>`_. It provides MARL research community with a unified platform for building, training, and evaluating MARL algorithms.

.. figure:: ../images/marllib_open.png
    :align: center

    Overview of the MARLlib architecture.


There are five core features of **MARLlib**.

- It unifies multi-agent environment interfaces with a new interface following Gym standard and supports both synchronous and asynchronous agent-environment interaction. Currently, MARLlib provides support to ten environments.
- It unifies diverse algorithm pipeline with a newly proposed single-agent perspective of implementation. Currently, MARLlib incorporates 18 algorithms and is able to handle cooperative (team-reward-only cooperation), collaborative (individual-reward-accessible cooperation), competitive (individual competition), and mixed (teamwork-based competition) tasks.
- It classifies algorithms into independent learning, centralized critic, and value decomposition categories(inspired by EPyMARL) and enables module reuse and extensibility within each category.
- It provides three parameter sharing strategies, namely full-sharing, non-sharing, and group-sharing, by implementing the policy mapping API of RLlib. This is implemented to be fully decoupled from algorithms and environments, and is completely controlled by configuration files.
- It provides standard 2 or 20 millions timesteps learning curve in the form of CSV of each task-algorithm for reference. These results are reproducible as configuration files for each experiment are provided along.

Before starting, please ensure you've installed the dependencies by following the :ref:`basic-installation`.
The environment-specific description is maintained in :ref:`env`.
:ref:`quick-start` gives some basic examples.

