A2C Family: IA2C, MAA2C, and VDA2C
======================================================================

.. contents::
    :local:
    :depth: 3

Read List
-------------

- `High-Dimensional Continuous Control Using Generalized Advantage Estimation <https://arxiv.org/abs/1506.02438>`_
- `Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_
- `Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge? <https://arxiv.org/abs/2011.09533>`_
- `The Surprising Effectiveness of A2C in Cooperative, Multi-Agent Games <https://arxiv.org/abs/2103.01955>`_


A recap of Proximal Policy Optimization
-----------------------------------------------

Preliminary
^^^^^^^^^^^^^^^

Algorithm Insights
^^^^^^^^^^^^^^^^^^^^^^^

Math Formulation
^^^^^^^^^^^^^^^^^^


IA2C: multi-agent version of A2C
-----------------------------------------------------

.. admonition:: Quick Facts

    - Independent proximal policy optimization is a natural extension of standard single-agent proximal policy optimization in multi-agent settings.
    - The sampling/training pipeline is the same when we stand at the view of a single agent when comparing A2C and IA2C.
    - Agent architecture of IA2C consists of two modules: policy network and critic network.
    - IA2C applies to cooperative, competitive, and mixed task modes.

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Vanilla Policy Gradient (PG) & Trust Region Policy Optimization (TRPO) & General Advantage Estimation (GAE)


Characteristic
^^^^^^^^^^^^^^^

action space

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``discrete``
     - ``continues``

task mode

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``cooperative``
     - ``collaborative``
     - ``competitive``

taxonomy label

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``on-policy``
     - ``stochastic``
     - ``independent learning``

derived algorithm

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - :ref:`MAA2C`
     - :ref:`HAA2C`
     - :ref:`VDA2C`


Algorithm Insights
^^^^^^^^^^^^^^^^^^^^^^^

A2C is a first-order optimization that simplifies its implementation. Similar to TRPO objective function, It defines the probability ratio between the new policy and old policy as :math:`\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}`.
Instead of adding complicated KL constraints, A2C imposes this policy ratio to stay within a small interval between :math:`1-\epsilon` and :math:`1+\epsilon`.
The objective function of A2C takes the minimum value between the original value and the clipped value.

There are two primary variants of A2C: A2C-Penalty and A2C-Clip. Here we only give the formulation of A2C-Clip, which is more commonly used in practice.

Math Formulation
^^^^^^^^^^^^^^^^^^

Critic learning:

.. math::

    \phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( V_{\phi} (s_t) - \hat{R}_t \right)^2

General Advantage Estimation:

.. math::

    A_t=\sum_{t=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^V


Policy learning:

.. math::

    L(s,a,\theta_k,\theta) = \min\left(
    \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\;
    \text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, 1 - \epsilon, 1+\epsilon \right) A^{\pi_{\theta_k}}(s,a)
    \right),

Here
:math:`{\mathcal D}` is the collected trajectories.
:math:`R` is the rewards-to-go.
:math:`\tau` is the trajectory.
:math:`V_{\phi}` is the critic function.
:math:`A` is the advantage.
:math:`\gamma` is discount value.
:math:`\lambda` is the weight value of GAE.
:math:`a` is the action.
:math:`s` is the observation/state.
:math:`\epsilon` is a hyperparameter controlling how far away the new policy is allowed to go from the old.
:math:`\pi_{\theta}` is the policy net.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In IA2C, each agent follows a standard A2C sampling/training pipeline. Therefore, IA2C is a general baseline for all MARL tasks with robust performance.

.. figure:: ../images/ippo.png
    :width: 600
    :align: center

    Independent Proximal Policy Optimization (IA2C)

Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We use vanilla A2C implementation of RLlib in IA2C. The only exception is we rewrite the SGD iteration logic.
The differences can be found in

    - ``MultiGPUTrainOneStep``
    - ``learn_on_loaded_batch``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/ppo``
- ``marl/algos/hyperparams/fintuned/env/ppo``

Usage & Limitation
^^^^^^^^^^^^^^^^^^^^^^

IA2C in *MARLlib* is suitable for

- continues control tasks
- discrete control tasks
- any task mode

.. code-block:: shell

    python marl/main.py --algo_config=ppo --finetuned --env-config=smac with env_args.map_name=3m



MAA2C: A2C agent with a centralized critic
-----------------------------------------------------

.. admonition:: Quick Facts

    - Multi-agent proximal policy optimization (MAA2C) is one of the centralized extensions of :ref:`IA2C`.
    - Agent architecture of MAA2C consists of two modules: policy network and critic network.
    - MAA2C outperforms other MARL algorithms in most multi-agent tasks, especially when agents are homogeneous.
    - MAA2C is proposed to solve cooperative tasks but is still applicable to collaborative, competitive, and mixed tasks.

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`IA2C`

Characteristic
^^^^^^^^^^^^^^^

action space

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``discrete``
     - ``continues``

task mode

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``cooperative``
     - ``collaborative``
     - ``competitive``

taxonomy label

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``on-policy``
     - ``stochastic``
     - ``centralized critic``

inherited algorithm

.. list-table::
   :widths: 25
   :header-rows: 0

   * - :ref:`IA2C`




Algorithm Insights
^^^^^^^^^^^^^^^^^^^^^^^

On-policy reinforcement learning algorithm is less utilized than off-policy learning algorithms in multi-agent settings.
This is often due to the belief that on-policy methods are less sample efficient than their off-policy counterparts in multi-agent problems.
The MAA2C paper proves that:

#. On-policy algorithms can achieve comparable performance to various off-policy methods.
#. MAA2C is a robust MARL algorithm for diverse cooperative tasks and can outperform SOTA off-policy methods in more challenging scenarios.
#. Formulating the input to the centralized value function is crucial for the final performance.
#. Tricks in MAA2C training are essential.

.. admonition:: Some Interesting Facts

    - MAA2C paper is done in cooperative settings. Nevertheless, it can be directly applied to competitive and mixed task modes. Moreover, the performance is still good.
    - MAA2C paper adopts some other tricks like death masking and clipping ratio. But compared to the input formulation, these tricks' impact is not so significant.
    - Sampling procedure of on-policy algorithms can be parallel conducted. Therefore, the actual time consuming for a comparable performance between on-policy and off-policy algorithms is almost the same when we have enough sampling *workers*.
    - The parameters are shared across agents. However, not sharing these parameters will not incur any problems. On the opposite, partly sharing these parameters(e.g., only sharing the critic) can help achieve better performance in some scenarios.


Math Formulation
^^^^^^^^^^^^^^^^^^

Critic learning:

.. math::

    \phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( V_{\phi} (s_t) - \hat{R}_t \right)^2

General Advantage Estimation:

.. math::

    A_t=\sum_{t=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^V


Policy learning:

.. math::

    L(s,\mathbf{s}^-, a,\mathbf{a}^-,\theta_k,\theta) = \min\left(
    \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s, \mathbf{s}^-,\mathbf{a}^-), \;\;
    \text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, 1 - \epsilon, 1+\epsilon \right) A^{\pi_{\theta_k}}(s, \mathbf{s}^-,\mathbf{a}^-)
    \right),

Here
:math:`{\mathcal D}` is the collected trajectories.
:math:`R` is the rewards-to-go.
:math:`\tau` is the trajectory.
:math:`A` is the advantage.
:math:`\gamma` is discount value.
:math:`\lambda` is the weight value of GAE.
:math:`a` is the current agent action.
:math:`\mathbf{a}^-` is the action set of all agents, except the current agent.
:math:`s` is the current agent observation/state.
:math:`\mathbf{s}^-` is the observation/state set of all agents, except the current agent.
:math:`\epsilon` is a hyperparameter controlling how far away the new policy is allowed to go from the old.
:math:`V_{\phi}` is the critic value function.
:math:`\pi_{\theta}` is the policy net.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the sampling stage, agents share information with others. The information includes others' observations and predicted actions. After collecting the necessary information from other agents,
all agents follow the standard A2C training pipeline, except using the centralized critic value function to calculate the GAE and conduct the A2C critic learning procedure.

.. figure:: ../images/mappo.png
    :width: 600
    :align: center

    Multi-agent Proximal Policy Optimization (MAA2C)

Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We use vanilla A2C implementation of RLlib in IA2C. The only exception is we rewrite the SGD iteration logic.
The differences can be found in

    - ``MultiGPUTrainOneStep``
    - ``learn_on_loaded_batch``

Based on IA2C, we add centralized modules to implement MAA2C.
The main differences are:

    - ``centralized_critic_postprocessing``
    - ``central_critic_ppo_loss``
    - ``CC_RNN``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/ppo``
- ``marl/algos/hyperparams/fintuned/env/ppo``

Usage & Limitation
^^^^^^^^^^^^^^^^^^^^^^

IA2C in *MARLlib* is suitable for

- continues control tasks
- discrete control tasks
- any task mode

.. code-block:: shell

    python marl/main.py --algo_config=ppo --finetuned --env-config=smac with env_args.map_name=3m




VDA2C: mixing the critic of a bunch of A2C agents
-----------------------------------------------------

.. admonition:: Quick Facts

    - Multi-agent proximal policy optimization (MAA2C) is one of the centralized extensions of :ref:`IA2C`.
    - Agent architecture of MAA2C consists of two modules: policy network and critic network.
    - MAA2C outperforms other MARL algorithms in most multi-agent tasks, especially when agents are homogeneous.
    - MAA2C is proposed to solve cooperative tasks but is still applicable to collaborative, competitive, and mixed tasks.

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`IA2C`

Characteristic
^^^^^^^^^^^^^^^

action space

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``discrete``
     - ``continues``

task mode

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``cooperative``
     - ``collaborative``
     - ``competitive``

taxonomy label

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``on-policy``
     - ``stochastic``
     - ``centralized critic``

inherited algorithm

.. list-table::
   :widths: 25
   :header-rows: 0

   * - :ref:`IA2C`




Algorithm Insights
^^^^^^^^^^^^^^^^^^^^^^^

On-policy reinforcement learning algorithm is less utilized than off-policy learning algorithms in multi-agent settings.
This is often due to the belief that on-policy methods are less sample efficient than their off-policy counterparts in multi-agent problems.
The MAA2C paper proves that:

#. On-policy algorithms can achieve comparable performance to various off-policy methods.
#. MAA2C is a robust MARL algorithm for diverse cooperative tasks and can outperform SOTA off-policy methods in more challenging scenarios.
#. Formulating the input to the centralized value function is crucial for the final performance.
#. Tricks in MAA2C training are essential.

.. admonition:: Some Interesting Facts

    - MAA2C paper is done in cooperative settings. Nevertheless, it can be directly applied to competitive and mixed task modes. Moreover, the performance is still good.
    - MAA2C paper adopts some other tricks like death masking and clipping ratio. But compared to the input formulation, these tricks' impact is not so significant.
    - Sampling procedure of on-policy algorithms can be parallel conducted. Therefore, the actual time consuming for a comparable performance between on-policy and off-policy algorithms is almost the same when we have enough sampling *workers*.
    - The parameters are shared across agents. However, not sharing these parameters will not incur any problems. On the opposite, partly sharing these parameters(e.g., only sharing the critic) can help achieve better performance in some scenarios.


Math Formulation
^^^^^^^^^^^^^^^^^^

Critic learning:

.. math::

    \phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( V_{\phi} (s_t) - \hat{R}_t \right)^2

General Advantage Estimation:

.. math::

    A_t=\sum_{t=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^V


Policy learning:

.. math::

    L(s,\mathbf{s}^-, a,\mathbf{a}^-,\theta_k,\theta) = \min\left(
    \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s, \mathbf{s}^-,\mathbf{a}^-), \;\;
    \text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, 1 - \epsilon, 1+\epsilon \right) A^{\pi_{\theta_k}}(s, \mathbf{s}^-,\mathbf{a}^-)
    \right),

Here
:math:`{\mathcal D}` is the collected trajectories.
:math:`R` is the rewards-to-go.
:math:`\tau` is the trajectory.
:math:`A` is the advantage.
:math:`\gamma` is discount value.
:math:`\lambda` is the weight value of GAE.
:math:`a` is the current agent action.
:math:`\mathbf{a}^-` is the action set of all agents, except the current agent.
:math:`s` is the current agent observation/state.
:math:`\mathbf{s}^-` is the observation/state set of all agents, except the current agent.
:math:`\epsilon` is a hyperparameter controlling how far away the new policy is allowed to go from the old.
:math:`V_{\phi}` is the critic value function.
:math:`\pi_{\theta}` is the policy net.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the sampling stage, agents share information with others. The information includes others' observations and predicted actions. After collecting the necessary information from other agents,
all agents follow the standard A2C training pipeline, except using the centralized critic value function to calculate the GAE and conduct the A2C critic learning procedure.

.. figure:: ../images/mappo.png
    :width: 600
    :align: center

    Multi-agent Proximal Policy Optimization (MAA2C)

Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We use vanilla A2C implementation of RLlib in IA2C. The only exception is we rewrite the SGD iteration logic.
The differences can be found in

    - ``MultiGPUTrainOneStep``
    - ``learn_on_loaded_batch``

Based on IA2C, we add centralized modules to implement MAA2C.
The main differences are:

    - ``centralized_critic_postprocessing``
    - ``central_critic_ppo_loss``
    - ``CC_RNN``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/ppo``
- ``marl/algos/hyperparams/fintuned/env/ppo``

Usage & Limitation
^^^^^^^^^^^^^^^^^^^^^^

IA2C in *MARLlib* is suitable for

- continues control tasks
- discrete control tasks
- any task mode

.. code-block:: shell

    python marl/main.py --algo_config=ppo --finetuned --env-config=smac with env_args.map_name=3m

