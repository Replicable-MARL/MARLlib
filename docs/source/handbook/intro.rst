.. _intro:

Introduction
============

**Multi-Agent RLlib (MARLlib)** is **a comprehensive Multi-Agent Reinforcement Learning algorithm library** based on `Ray <https://github.com/ray-project/ray>`_ and one of its toolkits `RLlib <https://github.com/ray-project/ray/tree/master/rllib>`_. It provides MARL research community with a unified platform for building, training, and evaluating MARL algorithms.

.. figure:: ../images/marllib_open.png
    :align: center

    Overview of the MARLlib architecture.


There are four core features of **MARLlib**.

- It unifies diverse algorithm pipeline with a newly proposed agent-level distributed dataflow. Currently, MARLlib delivers 18 algorithms and is able to handle cooperative (team-reward-only cooperation), collaborative (individual-reward-accessible cooperation), competitive (individual competition), and mixed (teamwork-based competition) tasks.
- It unifies multi-agent environment interfaces with a new interface following Gym standard and supports both synchronous and asynchronous agent-environment interaction. Currently, MARLlib provides support to ten environments.
- It provides three parameter sharing strategies, namely full-sharing, non-sharing, and group-sharing, by implementing the policy mapping API of RLlib. This is implemented to be fully decoupled from algorithms and environments, and is completely controlled by configuration files.
- It provides standard 2 or 20 millions timesteps learning curve in the form of CSV of each task-algorithm for reference. These results are reproducible as configuration files for each experiment are provided along. In total, more than a thousand experiments are conducted and released. 

Before starting, please ensure you've installed the dependencies by following the :ref:`basic-installation`.
The environment-specific description is maintained in :ref:`env`.
:ref:`quick-start` gives some basic examples.

Citing MARLlib
^^^^^^^^^^^^^^^^

If you use MARLlib in your work, please cite the accompanying `paper <https://arxiv.org/abs/2210.13708>`_.

.. code-block:: bibtex

    @article{hu2022marllib,
      title={MARLlib: Extending RLlib for Multi-agent Reinforcement Learning},
      author={Hu, Siyi and Zhong, Yifan and Gao, Minquan and Wang, Weixun and Dong, Hao and Li, Zhihui and Liang, Xiaodan and Chang, Xiaojun and Yang, Yaodong},
      journal={arXiv preprint arXiv:2210.13708},
      year={2022}
    }

