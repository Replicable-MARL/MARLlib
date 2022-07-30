Trust Region Policy Optimization Family
======================================================================

.. contents::
    :local:
    :depth: 3

Read List
-------------

- `High-Dimensional Continuous Control Using Generalized Advantage Estimation <https://arxiv.org/abs/1506.02438>`_
- `Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_
- `Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge? <https://arxiv.org/abs/2011.09533>`_
- `The Surprising Effectiveness of TRPO in Cooperative, Multi-Agent Games <https://arxiv.org/abs/2103.01955>`_


A recap of Trust Region Policy Optimization
-----------------------------------------------

Preliminary
^^^^^^^^^^^^^^^

Algorithm Insights
^^^^^^^^^^^^^^^^^^^^^^^

Math Formulation
^^^^^^^^^^^^^^^^^^


ITRPO: multi-agent version of TRPO
-----------------------------------------------------

.. admonition:: Quick Facts

    - Independent proximal policy optimization is a natural extension of standard single-agent proximal policy optimization in multi-agent settings.
    - The sampling/training pipeline is the same when we stand at the view of a single agent when comparing TRPO and ITRPO.
    - Agent architecture of ITRPO consists of two modules: policy network and critic network.
    - ITRPO applies to cooperative, competitive, and mixed task modes.

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

   * - :ref:`MATRPO`
     - :ref:`HATRPO`
     - :ref:`VDTRPO`


Algorithm Insights
^^^^^^^^^^^^^^^^^^^^^^^

TRPO is a first-order optimization that simplifies its implementation. Similar to TRPO objective function, It defines the probability ratio between the new policy and old policy as :math:`\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}`.
Instead of adding complicated KL constraints, TRPO imposes this policy ratio to stay within a small interval between :math:`1-\epsilon` and :math:`1+\epsilon`.
The objective function of TRPO takes the minimum value between the original value and the clipped value.

There are two primary variants of TRPO: TRPO-Penalty and TRPO-Clip. Here we only give the formulation of TRPO-Clip, which is more commonly used in practice.

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

In ITRPO, each agent follows a standard TRPO sampling/training pipeline. Therefore, ITRPO is a general baseline for all MARL tasks with robust performance.

.. figure:: ../images/iTRPO.png
    :width: 600
    :align: center

    Independent Proximal Policy Optimization (ITRPO)

Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We use vanilla TRPO implementation of RLlib in ITRPO. The only exception is we rewrite the SGD iteration logic.
The differences can be found in

    - ``MultiGPUTrainOneStep``
    - ``learn_on_loaded_batch``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/TRPO``
- ``marl/algos/hyperparams/fintuned/env/TRPO``

Usage & Limitation
^^^^^^^^^^^^^^^^^^^^^^

ITRPO in *MARLlib* is suitable for

- continues control tasks
- discrete control tasks
- any task mode

.. code-block:: shell

    python marl/main.py --algo_config=TRPO --finetuned --env-config=smac with env_args.map_name=3m



MATRPO: TRPO agent with a centralized critic
-----------------------------------------------------

.. admonition:: Quick Facts

    - Multi-agent proximal policy optimization (MATRPO) is one of the centralized extensions of :ref:`ITRPO`.
    - Agent architecture of MATRPO consists of two modules: policy network and critic network.
    - MATRPO outperforms other MARL algorithms in most multi-agent tasks, especially when agents are homogeneous.
    - MATRPO is proposed to solve cooperative tasks but is still applicable to collaborative, competitive, and mixed tasks.

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`ITRPO`

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

   * - :ref:`ITRPO`




Algorithm Insights
^^^^^^^^^^^^^^^^^^^^^^^

On-policy reinforcement learning algorithm is less utilized than off-policy learning algorithms in multi-agent settings.
This is often due to the belief that on-policy methods are less sample efficient than their off-policy counterparts in multi-agent problems.
The MATRPO paper proves that:

#. On-policy algorithms can achieve comparable performance to various off-policy methods.
#. MATRPO is a robust MARL algorithm for diverse cooperative tasks and can outperform SOTA off-policy methods in more challenging scenarios.
#. Formulating the input to the centralized value function is crucial for the final performance.
#. Tricks in MATRPO training are essential.

.. admonition:: Some Interesting Facts

    - MATRPO paper is done in cooperative settings. Nevertheless, it can be directly applied to competitive and mixed task modes. Moreover, the performance is still good.
    - MATRPO paper adopts some other tricks like death masking and clipping ratio. But compared to the input formulation, these tricks' impact is not so significant.
    - Sampling procedure of on-policy algorithms can be parallel conducted. Therefore, the actual time consuming for a comparable performance between on-policy and off-policy algorithms is almost the same when we have enough sampling *workers*.
    - The parameters are shared across agents. However, not sharing these parameters will not incur any problems. On the oTRPOsite, partly sharing these parameters(e.g., only sharing the critic) can help achieve better performance in some scenarios.


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
all agents follow the standard TRPO training pipeline, except using the centralized critic value function to calculate the GAE and conduct the TRPO critic learning procedure.

.. figure:: ../images/maTRPO.png
    :width: 600
    :align: center

    Multi-agent Proximal Policy Optimization (MATRPO)

Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We use vanilla TRPO implementation of RLlib in ITRPO. The only exception is we rewrite the SGD iteration logic.
The differences can be found in

    - ``MultiGPUTrainOneStep``
    - ``learn_on_loaded_batch``

Based on ITRPO, we add centralized modules to implement MATRPO.
The main differences are:

    - ``centralized_critic_postprocessing``
    - ``central_critic_TRPO_loss``
    - ``CC_RNN``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/TRPO``
- ``marl/algos/hyperparams/fintuned/env/TRPO``

Usage & Limitation
^^^^^^^^^^^^^^^^^^^^^^

ITRPO in *MARLlib* is suitable for

- continues control tasks
- discrete control tasks
- any task mode

.. code-block:: shell

    python marl/main.py --algo_config=TRPO --finetuned --env-config=smac with env_args.map_name=3m




VDTRPO: mixing the critic of a bunch of TRPO agents
-----------------------------------------------------

.. admonition:: Quick Facts

    - Multi-agent proximal policy optimization (MATRPO) is one of the centralized extensions of :ref:`ITRPO`.
    - Agent architecture of MATRPO consists of two modules: policy network and critic network.
    - MATRPO outperforms other MARL algorithms in most multi-agent tasks, especially when agents are homogeneous.
    - MATRPO is proposed to solve cooperative tasks but is still applicable to collaborative, competitive, and mixed tasks.

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`ITRPO`

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

   * - :ref:`ITRPO`




Algorithm Insights
^^^^^^^^^^^^^^^^^^^^^^^

On-policy reinforcement learning algorithm is less utilized than off-policy learning algorithms in multi-agent settings.
This is often due to the belief that on-policy methods are less sample efficient than their off-policy counterparts in multi-agent problems.
The MATRPO paper proves that:

#. On-policy algorithms can achieve comparable performance to various off-policy methods.
#. MATRPO is a robust MARL algorithm for diverse cooperative tasks and can outperform SOTA off-policy methods in more challenging scenarios.
#. Formulating the input to the centralized value function is crucial for the final performance.
#. Tricks in MATRPO training are essential.

.. admonition:: Some Interesting Facts

    - MATRPO paper is done in cooperative settings. Nevertheless, it can be directly applied to competitive and mixed task modes. Moreover, the performance is still good.
    - MATRPO paper adopts some other tricks like death masking and clipping ratio. But compared to the input formulation, these tricks' impact is not so significant.
    - Sampling procedure of on-policy algorithms can be parallel conducted. Therefore, the actual time consuming for a comparable performance between on-policy and off-policy algorithms is almost the same when we have enough sampling *workers*.
    - The parameters are shared across agents. However, not sharing these parameters will not incur any problems. On the oTRPOsite, partly sharing these parameters(e.g., only sharing the critic) can help achieve better performance in some scenarios.


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
all agents follow the standard TRPO training pipeline, except using the centralized critic value function to calculate the GAE and conduct the TRPO critic learning procedure.

.. figure:: ../images/maTRPO.png
    :width: 600
    :align: center

    Multi-agent Proximal Policy Optimization (MATRPO)

Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We use vanilla TRPO implementation of RLlib in ITRPO. The only exception is we rewrite the SGD iteration logic.
The differences can be found in

    - ``MultiGPUTrainOneStep``
    - ``learn_on_loaded_batch``

Based on ITRPO, we add centralized modules to implement MATRPO.
The main differences are:

    - ``centralized_critic_postprocessing``
    - ``central_critic_TRPO_loss``
    - ``CC_RNN``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/TRPO``
- ``marl/algos/hyperparams/fintuned/env/TRPO``

Usage & Limitation
^^^^^^^^^^^^^^^^^^^^^^

ITRPO in *MARLlib* is suitable for

- continues control tasks
- discrete control tasks
- any task mode

.. code-block:: shell

    python marl/main.py --algo_config=TRPO --finetuned --env-config=smac with env_args.map_name=3m


HATRPO: Sequentially updating critic of MATRPO agents
-----------------------------------------------------

.. admonition:: Quick Facts

    - Multi-agent proximal policy optimization (MATRPO) is one of the centralized extensions of :ref:`ITRPO`.
    - Agent architecture of MATRPO consists of two modules: policy network and critic network.
    - MATRPO outperforms other MARL algorithms in most multi-agent tasks, especially when agents are homogeneous.
    - MATRPO is proposed to solve cooperative tasks but is still applicable to collaborative, competitive, and mixed tasks.

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`ITRPO`

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

   * - :ref:`ITRPO`




Algorithm Insights
^^^^^^^^^^^^^^^^^^^^^^^

On-policy reinforcement learning algorithm is less utilized than off-policy learning algorithms in multi-agent settings.
This is often due to the belief that on-policy methods are less sample efficient than their off-policy counterparts in multi-agent problems.
The MATRPO paper proves that:

#. On-policy algorithms can achieve comparable performance to various off-policy methods.
#. MATRPO is a robust MARL algorithm for diverse cooperative tasks and can outperform SOTA off-policy methods in more challenging scenarios.
#. Formulating the input to the centralized value function is crucial for the final performance.
#. Tricks in MATRPO training are essential.

.. admonition:: Some Interesting Facts

    - MATRPO paper is done in cooperative settings. Nevertheless, it can be directly applied to competitive and mixed task modes. Moreover, the performance is still good.
    - MATRPO paper adopts some other tricks like death masking and clipping ratio. But compared to the input formulation, these tricks' impact is not so significant.
    - Sampling procedure of on-policy algorithms can be parallel conducted. Therefore, the actual time consuming for a comparable performance between on-policy and off-policy algorithms is almost the same when we have enough sampling *workers*.
    - The parameters are shared across agents. However, not sharing these parameters will not incur any problems. On the oTRPOsite, partly sharing these parameters(e.g., only sharing the critic) can help achieve better performance in some scenarios.


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
all agents follow the standard TRPO training pipeline, except using the centralized critic value function to calculate the GAE and conduct the TRPO critic learning procedure.

.. figure:: ../images/maTRPO.png
    :width: 600
    :align: center

    Multi-agent Proximal Policy Optimization (MATRPO)

Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We use vanilla TRPO implementation of RLlib in ITRPO. The only exception is we rewrite the SGD iteration logic.
The differences can be found in

    - ``MultiGPUTrainOneStep``
    - ``learn_on_loaded_batch``

Based on ITRPO, we add centralized modules to implement MATRPO.
The main differences are:

    - ``centralized_critic_postprocessing``
    - ``central_critic_TRPO_loss``
    - ``CC_RNN``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/TRPO``
- ``marl/algos/hyperparams/fintuned/env/TRPO``

Usage & Limitation
^^^^^^^^^^^^^^^^^^^^^^

ITRPO in *MARLlib* is suitable for

- continues control tasks
- discrete control tasks
- any task mode

.. code-block:: shell

    python marl/main.py --algo_config=TRPO --finetuned --env-config=smac with env_args.map_name=3m