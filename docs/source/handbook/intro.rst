Introduction
============

Multi-Agent RLlib (MARLlib) is a MARL benchmark based on Ray and one of its toolkits, RLlib. It provides the MARL research community a unified platform for developing and evaluating new ideas in various multi-agent environments.

.. figure:: ../images/marllib_open.png
    :align: center

    Overview of the MARLlib architecture.


The key features of MARLlib include:

* it collects most of the existing MARL algorithms widely acknowledged by the community and unifies them under one framework.
* it gives a solution that enables different multi-agent environments using the same interface to interact with the agents.
* it guarantees excellent efficiency in both the training and sampling process.
* it provides trained results, including learning curves and pretrained models specific to each task and algorithm's combination, with finetuned hyper-parameters to guarantee credibility.

Before starting, please ensure you've installed the fundamentally required dependency by following the :ref:`basic-installation`.
The environment-specific description is maintained in :ref:`env`.
:ref:`quick-start` gives some basic examples.

