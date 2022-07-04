Introduction
============

MARLlib is a parallel framework of population-based learning nested with (multi-agent) reinforcement learning (RL) methods, such as Policy Space Response Oracle, Self-Play, and Neural Fictitious Self-Play. MARLlib provides higher-level abstractions of MARL training paradigms, which enables efficient code reuse and flexible deployments on different distributed computing paradigms. The design of MARLlib also strives to promote the research of other multi-agent learning research, including multi-agent imitation learning and model-based RL.

.. figure:: ../imgs/overview.png
    :align: center

    Overview of the MARLlib architecture.


The key features of MARLlib include:

* **Pytorch-based algorithm implementation**: All algorithms implemented in MARLlib are based on `PyTorch <https://pytorch.org/>`_.
* **Popular distributed computing RL framework support**: MARLlib support multiple distributed computing RL frameworks, including asynchronous
* **Provide comprehensive multi-agent RL training interfaces**: MARLlib implemented abstractions of several popular MARL training paradigms, aims to accelerate the development of algorithms on, and make users focus on the development of core algorithms, not training flow customization.


Before getting start, please make sure you've installed MARLlib by following the :ref:`installation`. :ref:`quick-start` gives some basic examples. As for the API documentation.

Citing MARLlib
^^^^^^^^^^^^
