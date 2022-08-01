Proximal Policy Optimization Family
======================================================================

.. contents::
    :local:
    :depth: 3


.. _PPO:

PPO: A Recap
-----------------------------------------------

Preliminary
^^^^^^^^^^^^^^^

Vanilla Policy Gradient (PG) & Trust Region Policy Optimization (TRPO) & General Advantage Estimation (GAE)


Algorithm Insights
^^^^^^^^^^^^^^^^^^^^^^^

PPO is a first-order optimization that simplifies its implementation. Similar to TRPO objective function, It defines the probability ratio between the new policy and old policy as :math:`\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}`.
Instead of adding complicated KL constraints, PPO imposes this policy ratio to stay within a small interval between :math:`1-\epsilon` and :math:`1+\epsilon`.
The objective function of PPO takes the minimum value between the original value and the clipped value.

There are two primary variants of PPO: PPO-Penalty and PPO-Clip. Here we only give the formulation of PPO-Clip, which is more commonly used in practice.
For PPO-penalty, please refer to `Proximal Policy Optimization <https://spinningup.openai.com/en/latest/algorithms/ppo.html>`_.

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



IPPO: multi-agent version of PPO
-----------------------------------------------------

.. admonition:: Quick Facts

    - Independent proximal policy optimization is a natural extension of standard single-agent proximal policy optimization in multi-agent settings.
    - The sampling/training pipeline is the same when we stand at the view of a single agent when comparing PPO and IPPO.
    - Agent architecture of IPPO consists of two modules: policy network and critic network.
    - IPPO applies to cooperative, competitive, and mixed task modes.

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`PPO`

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


Algorithm Insights
^^^^^^^^^^^^^^^^^^^^^^^

Independent Proximal Policy Optimization (IPPO) is the multi-agent version of standard PPO. Each agent is now a PPO-based sampler and learner.
IPPO has no need for information sharing including real/sampled data and predicted data.
While the knowledge sharing across agents is optional in IPPO.
Note: There is a discussion of information sharing concept here: :ref:`yousn`

Math Formulation
^^^^^^^^^^^^^^^^^^

Standing at the view of a single agent under multi-agent settings, the math formulation of IPPO is same as :ref:`PPO`.

Note in multi-agent settings, all the agent models can be shared including:

- :math:`V_{\phi}` is the critic net.
- :math:`\pi_{\theta}` is the policy net.

Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In IPPO, each agent follows a standard PPO sampling/training pipeline. Therefore, IPPO is a general baseline for all MARL tasks with robust performance.

.. figure:: ../images/ippo.png
    :width: 600
    :align: center

    Independent Proximal Policy Optimization (IPPO)

Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We use vanilla PPO implementation of RLlib in IPPO. The only exception is we rewrite the SGD iteration logic.
The differences can be found in

    - ``MultiGPUTrainOneStep``
    - ``learn_on_loaded_batch``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/ppo``
- ``marl/algos/hyperparams/fintuned/env/ppo``

Usage & Limitation
^^^^^^^^^^^^^^^^^^^^^^

IPPO in *MARLlib* is suitable for

- continues control tasks
- discrete control tasks
- any task mode

.. code-block:: shell

    python marl/main.py --algo_config=ppo --finetuned --env-config=smac with env_args.map_name=3m



MAPPO: PPO agent with a centralized critic
-----------------------------------------------------

.. admonition:: Quick Facts

    - Multi-agent proximal policy optimization (MAPPO) is one of the centralized extensions of :ref:`IPPO`.
    - Agent architecture of MAPPO consists of two modules: policy network and critic network.
    - MAPPO needs one stage information sharing on real/sampled data.
    - MAPPO is proposed to solve cooperative tasks but is still applicable to collaborative, competitive, and mixed tasks.

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`IPPO`

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

   * - :ref:`IPPO`




Algorithm Insights
^^^^^^^^^^^^^^^^^^^^^^^

On-policy reinforcement learning algorithm is less utilized than off-policy learning algorithms in multi-agent settings.
This is often due to the belief that on-policy methods are less sample efficient than their off-policy counterparts in multi-agent problems.
The MAPPO paper proves that:

#. On-policy algorithms can achieve comparable performance to various off-policy methods.
#. MAPPO is a robust MARL algorithm for diverse cooperative tasks and can outperform SOTA off-policy methods in more challenging scenarios.
#. Formulating the input to the centralized value function is crucial for the final performance.
#. Tricks in MAPPO training are essential.

.. admonition:: Some Interesting Facts

    - MAPPO paper is done in cooperative settings. Nevertheless, it can be directly applied to competitive and mixed task modes. Moreover, the performance is still good.
    - MAPPO paper adopts some other tricks like death masking and clipping ratio. But compared to the input formulation, these tricks' impact is not so significant.
    - Sampling procedure of on-policy algorithms can be parallel conducted. Therefore, the actual time consuming for a comparable performance between on-policy and off-policy algorithms is almost the same when we have enough sampling *workers*.
    - The parameters are shared across agents. However, not sharing these parameters will not incur any problems. On the opposite, partly sharing these parameters(e.g., only sharing the critic) can help achieve better performance in some scenarios.


Math Formulation
^^^^^^^^^^^^^^^^^^

MAPPO needs information sharing across agents. The critic learning utilize both self-observation and information provided by other agents including
 observation and actions. Here we bold the symbol (e.g., :math:`s` to :math:`\mathbf{s}`) to indicate more than one agent information is contained.

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
:math:`{\mathcal D}` is the collected trajectories, which can be shared across agents.
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
:math:`V_{\phi}` is the critic value function, which can be shared across agents.
:math:`\pi_{\theta}` is the policy net, which can be shared across agents.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the sampling stage, agents share information with others. The information includes others' observations and predicted actions. After collecting the necessary information from other agents,
all agents follow the standard PPO training pipeline, except using the centralized critic value function to calculate the GAE and conduct the PPO critic learning procedure.

.. figure:: ../images/mappo.png
    :width: 600
    :align: center

    Multi-agent Proximal Policy Optimization (MAPPO)

Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

Based on IPPO, we add centralized modules to implement MAPPO.
The main differences are:

    - ``centralized_critic_postprocessing``
    - ``central_critic_ppo_loss``
    - ``CC_RNN``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/mappo``
- ``marl/algos/hyperparams/fintuned/env/mappo``

Usage & Limitation
^^^^^^^^^^^^^^^^^^^^^^

IPPO in *MARLlib* is suitable for

- continues control tasks
- discrete control tasks
- any task mode

.. code-block:: shell

    python marl/main.py --algo_config=ppo --finetuned --env-config=smac with env_args.map_name=3m




VDPPO: mixing the critic of a bunch of PPO agents
-----------------------------------------------------

.. admonition:: Quick Facts

    - Value decomposition proximal policy optimization (VDPPO) is one of extensions of :ref:`IPPO`.
    - Agent architecture of VDPPO consists of three modules: policy network, critic network, and the mixer.
    - VDPPO is the algorithms combined QMIX, VDA2C, and, PPO.
    - VDPPO needs one stage of information sharing on real/sampled data and predicted data.
    - VDPPO is proposed to solve cooperative tasks only.

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`IPPO`

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
   :widths: 25
   :header-rows: 0

   * - ``cooperative``


taxonomy label

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``on-policy``
     - ``stochastic``
     - ``value decomposition``



Algorithm Insights
^^^^^^^^^^^^^^^^^^^^^^^

Value decomposition PPO focus on learning the credit assignment similar joint Q learning family, and belong to on-policy value decomposition algorithm.
VDPPO is easy to understand when you are familiar with both :ref:`QMIX` and :ref:`VDA2C`.

#. Like joint Q learning family, VDPPO is only applicable for cooperative multi-agent tasks.
#. The sample efficiency of VDPPO is worse than algorithms of joint Q learning family.
#. VDPPO can be applied to both discrete and continues control problem.

Math Formulation
^^^^^^^^^^^^^^^^^^

VDPPO needs information sharing across agents. The critic mixing utilizes both self-observation and other agents observation.
Here we bold the symbol (e.g., :math:`s` to :math:`\mathbf{s}`) to indicate more than one agent information is contained.


Critic mixing:

.. math::

    V_{tot}(\mathbf{a}, s;\boldsymbol{\phi},\psi) = g_{\psi}\bigl(`\mathbf{s}, V_{\phi_1},Q_{\phi_2},..,Q_{\phi_n} \bigr)



Critic learning:

.. math::

    \phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( V_{tot}(\mathbf{a}, s;\boldsymbol{\phi},\psi) - \hat{R}_t \right)^2

General Advantage Estimation:

.. math::

    A_t=\sum_{t=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^V_{tot}


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
:math:`g_{\psi}` is mixing network.

Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the sampling stage, agents share information with others. The information includes others' observations and predicted critic value. After collecting the necessary information from other agents,
all agents follow the standard PPO training pipeline, except using the mixed critic value to calculate the GAE and conduct the PPO critic learning procedure.

.. figure:: ../images/vdppo.png
    :width: 600
    :align: center

    Multi-agent Proximal Policy Optimization (MAPPO)

Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

Based on IPPO, we add mixing Q modules to implement VDPPO.
The main differences are:

    - ``value_mixing_postprocessing``
    - ``value_mix_ppo_surrogate_loss``
    - ``VD_RNN``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/vdppo``
- ``marl/algos/hyperparams/fintuned/env/vdppo``

Usage & Limitation
^^^^^^^^^^^^^^^^^^^^^^

IPPO in *MARLlib* is suitable for

- continues control tasks
- discrete control tasks
- any task mode

.. code-block:: shell

    python marl/main.py --algo_config=vdppo --finetuned --env-config=smac with env_args.map_name=3m


HAPPO: Sequentially updating critic of MAPPO agents
-----------------------------------------------------

.. admonition:: Quick Facts

    - Multi-agent proximal policy optimization (MAPPO) is one of the centralized extensions of :ref:`IPPO`.
    - Agent architecture of MAPPO consists of two modules: policy network and critic network.
    - MAPPO outperforms other MARL algorithms in most multi-agent tasks, especially when agents are homogeneous.
    - MAPPO is proposed to solve cooperative tasks but is still applicable to collaborative, competitive, and mixed tasks.

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`IPPO`

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

   * - :ref:`IPPO`




Algorithm Insights
^^^^^^^^^^^^^^^^^^^^^^^

On-policy reinforcement learning algorithm is less utilized than off-policy learning algorithms in multi-agent settings.
This is often due to the belief that on-policy methods are less sample efficient than their off-policy counterparts in multi-agent problems.
The MAPPO paper proves that:

#. On-policy algorithms can achieve comparable performance to various off-policy methods.
#. MAPPO is a robust MARL algorithm for diverse cooperative tasks and can outperform SOTA off-policy methods in more challenging scenarios.
#. Formulating the input to the centralized value function is crucial for the final performance.
#. Tricks in MAPPO training are essential.

.. admonition:: Some Interesting Facts

    - MAPPO paper is done in cooperative settings. Nevertheless, it can be directly applied to competitive and mixed task modes. Moreover, the performance is still good.
    - MAPPO paper adopts some other tricks like death masking and clipping ratio. But compared to the input formulation, these tricks' impact is not so significant.
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
all agents follow the standard PPO training pipeline, except using the centralized critic value function to calculate the GAE and conduct the PPO critic learning procedure.

.. figure:: ../images/mappo.png
    :width: 600
    :align: center

    Multi-agent Proximal Policy Optimization (MAPPO)

Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We use vanilla PPO implementation of RLlib in IPPO. The only exception is we rewrite the SGD iteration logic.
The differences can be found in

    - ``MultiGPUTrainOneStep``
    - ``learn_on_loaded_batch``

Based on IPPO, we add centralized modules to implement MAPPO.
The main differences are:

    - ``centralized_critic_postprocessing``
    - ``central_critic_ppo_loss``
    - ``CC_RNN``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/ppo``
- ``marl/algos/hyperparams/fintuned/env/ppo``

Usage & Limitation
^^^^^^^^^^^^^^^^^^^^^^

IPPO in *MARLlib* is suitable for

- continues control tasks
- discrete control tasks
- any task mode

.. code-block:: shell

    python marl/main.py --algo_config=ppo --finetuned --env-config=smac with env_args.map_name=3m

Read List
-------------

- `High-Dimensional Continuous Control Using Generalized Advantage Estimation <https://arxiv.org/abs/1506.02438>`_
- `Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_
- `Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge? <https://arxiv.org/abs/2011.09533>`_
- `The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games <https://arxiv.org/abs/2103.01955>`_
- `Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning <https://arxiv.org/abs/2109.11251>`_