Deep Deterministic Policy Gradient Family
======================================================================


.. contents::
    :local:
    :depth: 3

---------------------

.. _DDPG:

Deep Deterministic Policy Gradient: A Recap
-----------------------------------------------


**Preliminary**

- Q-Learning & Deep Q Network(DQN)

Deep Deterministic Policy Gradient (DDPG) is an algorithm that concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function and the Q-function to learn the policy.
The motivation of DDPG is to tackle the problem that standard Q-learning can only be used in discrete action space (a finite number of actions).
To extend the Q function to the continuous control problem, DDPG adopts an extra policy network :math:`\mu(s)` parameterized by :math:`\theta` to produce action value.
The Q value is estimated as :math:`Q(s,\mu(s))`. The Q function is parameterized by :math:`\phi`.

**Mathematical Form**

Q learning:

.. math::

    L(\phi, {\mathcal D}) = \underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[
        \Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) Q_{\phi_{\text{targ}}}(s', \mu_{\theta_{\text{targ}}}(s')) \right) \Bigg)^2
        \right]

Policy learning:

.. math::

    \max_{\theta} \underset{s \sim {\mathcal D}}{{\mathrm E}}\left[ Q_{\phi}(s, \mu_{\theta}(s)) \right]

Here :math:`{\mathcal D}` is the replay buffer
:math:`a` is the action taken.
:math:`r` is the reward.
:math:`s` is the observation/state.
:math:`s'` is the next observation/state.
:math:`d` is set to 1 (True) when episode ends else 0 (False).
:math:`{\gamma}` is discount value.
:math:`\mu_{\theta}` is policy net.
:math:`Q_{\phi}` is Q net.
:math:`\mu_{\theta_{\text{targ}}}` is target policy net
:math:`Q_{\phi_{\text{targ}}}` is target Q net.

.. admonition:: You Should Know

    Some tricks like `gumble softmax` enables DDPG policy net to output categorical-like action distribution.

---------------------

.. _IDDPG:

IDDPG: multi-agent version of DDPG
-------------------------------------

- Independent deep deterministic policy gradient (IDDPG) is a natural extension of standard deep deterministic policy gradient (DDPG) under multi-agent settings.
- The sampling/training pipeline is the same when we stand at the view of a single agent when comparing DDPG and IDDPG.
- An IDDPG agent architecture consists of two models: ``policy`` and ``Q``.
- IDDPG applies to cooperative, competitive, and mixed task modes.

**Preliminary**

- :ref:`DDPG`

Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each agent follows the standard DDPG learning pipeline. Models and Buffers can be shared or separated according to agents' group.
Note that buffer and agent models can be shared or separately training across agents. And this applies to all algorithms in DDPG family.

.. figure:: ../images/iddpg.png
    :width: 600
    :align: center

    Independent Deep Deterministic Policy Gradient (IDDPG)


Characteristic
^^^^^^^^^^^^^^^

action space

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``continues``

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

   * - ``off-policy``
     - ``deterministic``
     - ``independent learning``


Insights
^^^^^^^^^^^^^^^^^^^^^^^


Independent Deep Deterministic Policy Gradient (IDDPG) is the multi-agent version of standard DDPG. Each agent is now a DDPG-based sampler and learner.
IDDPG does not need information sharing, including real/sampled data and predicted data.
While knowledge sharing across agents is optional in IDDPG.

.. admonition:: Information Sharing

    In multi-agent learning, the concept of information sharing is not well defined and may confuse.
    Here we try to clarify this by categorizing the type of information sharing into three.

    - real/sampled data: observation, action, etc.
    - predicted data: Q/critic value, message for communication, etc.
    - knowledge: experience replay buffer, model parameters, etc.

    Knowledge-level information sharing is usually excluded from information sharing and is only seen as a trick.
    But recent works find it is essential for good performance. So here, we include knowledge sharing as part of the information sharing.


Mathematical Form
^^^^^^^^^^^^^^^^^^

Standing at the view of a single agent, the mathematical formulation of IDDPG is the same as DDPG: :ref:`DDPG`.

Note in multi-agent settings, all the agent models and buffer can be shared, including:

- :math:`{\mathcal D}` replay buffer.
- :math:`\mu_{\theta}` policy net.
- :math:`Q_{\phi}` Q net.
- :math:`\mu_{\theta_{\text{targ}}}` target policy net.
- :math:`Q_{\phi_{\text{targ}}}` target Q net.



Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We extend the vanilla IDDPG of RLlib to be recurrent neural network(RNN) compatible.
The main differences are:

- model side: the agent model-related modules and functions are rewritten, including:
    - ``build_rnnddpg_models_and_action_dist``
    - ``DDPG_RNN_TorchModel``
- algorithm side: the sampling and training pipelines are rewritten, including:
    - ``episode_execution_plan``
    - ``ddpg_actor_critic_loss``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/ddpg``
- ``marl/algos/hyperparams/fintuned/env/ddpg``

.. admonition:: Continues Control Tasks

    - There is only a few MARL dataset focusing on continuous control. The popular three are:
        - :ref:`MPE` (discrete+continues)
        - :ref:`MaMujoco` (continues only)
        - :ref:`MetaDrive` (continues only)

---------------------

.. _MADDPG:

MADDPG: DDPG agent with a centralized Q
--------------------------------------------


- Multi-agent deep deterministic policy gradient(MADDPG) is one of the extended version of :ref:`IDDPG`.
- Agent architecture of MADDPG consists of two models: ``policy`` and ``Q``.
- MADDPG needs two stages of information sharing on real/sampled data and predicted data.
- MADDPG applies to cooperative, competitive, and mixed task modes.

**Preliminary**

-:ref:`IDDPG`

Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the sampling stage, each agent follows the standard DDPG learning pipeline to infer the action but uses a centralized Q function to compute the Q value, which needs data sharing
before sending all the collected data to the buffer.
In the learning stage, each agent predicts its next action using the target policy and shares it with other agents before entering the training loop.

.. figure:: ../images/maddpg.png
    :width: 600
    :align: center

    Multi-agent Deep Deterministic Policy Gradient (MADDPG)

Characteristic
^^^^^^^^^^^^^^^

action space

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``continues``

task mode

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``cooperative``
     - ``collaborative``
     - ``competitive``

taxonomy label

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``off-policy``
     - ``deterministic``


Insights
^^^^^^^^^^^^^^^^^^^^^^^



Traditional reinforcement learning approaches such as Q-Learning or policy gradient are poorly suited to multi-agent environments because:

#. Each agent's policy changes as training progress.
#. The environment becomes non-stationary from the perspective of any individual agent.
#. Deep Q-learning becomes unstable due to points 1 & 2.
#. Policy gradient methods suffer from high variance in the coordination of agents due to points 1 & 2.

Multi-agent Deep Deterministic Policy Gradient (MADDPG) is an algorithm that extends DDPG with a centralized Q function that takes observation and action from current agents and other agents. Like DDPG, MADDPG also has a policy network :math:`\mu(s)` parameterized by :math:`\theta` to produce action value.
While the centralized Q value is calculated as :math:`Q(\mathbf{s},\mu(\mathbf{s}))` and the Q network is parameterized by :math:`\phi`.
Note :math:`s` in policy network is the self-observation/state while :math:`\mathbf{s}` in centralized Q is the joint observation/state, which also includes the opponents.


.. admonition:: You Should Know

    - MADDPG is the most famous work that started MARL research under centralized training and decentralized execution(CTDE) these years.
    - Other works find that Q-learning-based algorithms can perform well under similar settings. E.g., :ref:`QMIX`.
    - Recent works prove that policy gradient methods can be directly applied to MARL and maintain good performance. E.g., :ref:`IPPO`
    - MADDPG is criticized for its unstable performance in recent MARL research.

Mathematical Form
^^^^^^^^^^^^^^^^^^

MADDPG needs information sharing across agents. The Q learning utilizes self-observation and information other agents provide, including
 observation and actions. Here we bold the symbol (e.g., :math:`s` to :math:`\mathbf{s}`) to indicate more than one agent information is contained.


Q learning:

.. math::

    L(\phi, {\mathcal D}) = \underset{(\mathbf{s},\mathbf{a},r,\mathbf{s'},d) \sim {\mathcal D}}{{\mathrm E}}\left[
        \Bigg( Q_{\phi}(\mathbf{s},\mathbf{a}) - \left(r + \gamma (1 - d) Q_{\phi_{\text{targ}}}(\mathbf{s'}, \mu_{\theta_{\text{targ}}}(\mathbf{s'})) \right) \Bigg)^2
        \right]


Policy learning:

.. math::

    \max_{\theta} \underset{s \sim {\mathcal D}}{{\mathrm E}}\left[ Q_{\phi}(s,\mathbf{a}, \mu_{\theta}(s)) \right]

Here :math:`{\mathcal D}` is the replay buffer and can be shared across agents.
:math:`\mathbf{a}` is an action set, including opponents.
:math:`r` is the reward.
:math:`\mathbf{s}` is the observation/state set, including opponents.
:math:`\mathbf{s'}` is the next observation/state set, including opponents.
:math:`d` is set to 1(True) when an episode ends else 0(False).
:math:`{\gamma}` is discount value.
:math:`\mu_{\theta}` is a policy net that can be shared across agents.
:math:`Q_{\phi}` is Q net, which can be shared across agents.
:math:`\mu_{\theta_{\text{targ}}}` is target policy net, which can be shared across agents.
:math:`Q_{\phi_{\text{targ}}}` is target Q net, which can be shared across agents.


Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We extend the vanilla DDPG of RLlib to be recurrent neural network(RNN) compatible.
Based on RNN compatible DDPG, we add the centralized sampling and training module to the original pipeline.
The main differences between IDDPG and MADDPG are:

- model side: the agent model-related modules and functions are built in a centralized style:
    - ``build_maddpg_models_and_action_dist``
    - ``MADDPG_RNN_TorchModel``
- algorithm side: the sampling and training pipelines are built in a centralized style:
    - ``centralized_critic_q``
    - ``central_critic_ddpg_loss``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/maddpg``
- ``marl/algos/hyperparams/fintuned/env/maddpg``


.. admonition:: You Should Know

    -The policy inference procedure of MADDPG is kept the same as IDDPG.
    -Some tricks like `gumble softmax` enables MADDPG to output categorical-like action distribution.

---------------------

.. _FACMAC:

FACMAC: mixing a bunch of DDPG agents
-------------------------------------------------------------


- Factored Multi-Agent Centralised Policy Gradients (FACMAC) is one of the extended version of :ref:`IDDPG`.
- Agent architecture of FACMAC consists of three models: ``policy``, ``Q``, and ``mixer``.
- FACMAC needs two stages of information sharing on real/sampled data and predicted data.
- FACMAC applies to cooperative task mode only.

**Preliminary**:


- :ref:`IDDPG`
- :ref:`QMIX`

Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each agent follows the standard DDPG learning pipeline in the sampling stage to infer and send the action to the Q function to get the Q value. Data like observation/state is shared among agents
before sending the sampled data to the buffer.
In the learning stage, each agent predicts its Q value using the Q function, the next action using the target policy,  and the next Q value using the target Q function.
Then each agent shares the predicted data with other agents before entering the training loop.

.. figure:: ../images/facmac.png
    :width: 600
    :align: center

    Factored Multi-Agent Centralised Policy Gradients (FACMAC)

Characteristic
^^^^^^^^^^^^^^^

action space

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``continues``

task mode

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``cooperative``

taxonomy label

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``off-policy``
     - ``deterministic``
     - ``value decomposition``



Insights
^^^^^^^^^^^^^^^^^^^^^^^

FACMAC is a variant of :ref:`IDDPG` in the value decomposition method and a counterpart of :ref:`MADDPG`.
The main contribution of FACMAC is:

#. MARL's first value decomposition method can deal with a continuous control problem.
#. Proposed with a multi-agent benchmark :ref:`MaMujoco` that focuses on continuous control with heterogeneous agents.
#. It can also be applied to discrete action space with tricks like `gumble softmax` and keep robust performance

Compared to existing methods, FACMAC:

- outperforms MADDPG and other baselines in both discrete and continuous action tasks.
- scales better as the number of agents (and/or actions) and the complexity of the task increases.
- proves that factoring the critic can better take advantage of our centralized gradient estimator to optimize the agent policies when the number of agents and/or actions is large.

.. admonition:: You Should Know

    - Recent works prove that stochastic policy gradient methods are more stable and perform well in tackling MARL. E.g., :ref:`MAA2C`. If you need better performance, try stochastic policy gradient methods.
    - Applicable scenarios of FACMAC are pretty restrained. E.g., the cooperative task only, the continuous task only(without adding tricks).


Mathematical Form
^^^^^^^^^^^^^^^^^^

MADDPG needs information sharing across agents. Therefore, the Q mixing utilizes both self-observation and other agents' observation.
Here we bold the symbol (e.g., :math:`s` to :math:`\mathbf{s}`) to indicate more than one agent information is contained.


Q mixing:

.. math::

    Q_{tot}(\mathbf{a}, s;\boldsymbol{\phi},\psi) = g_{\psi}\bigl(`\mathbf{s}, Q_{\phi_1},Q_{\phi_2},..,Q_{\phi_n} \bigr)

Q learning:

.. math::

    L(\phi,\psi, {\mathcal D}) = \underset{(\mathbf{s},\mathbf{a},r,\mathbf{s'},d) \sim {\mathcal D}}{{\mathrm E}}\left[
        \Bigg(Q_{tot}(\mathbf{a}, s;\boldsymbol{\phi},\psi) - \left(r + \gamma (1 - d) Q_{tot}(\mathbf{a'}, s';\boldsymbol{\phi_{\text{targ}}},\psi_{\text{targ}}) \right) \Bigg)^2
        \right]


Policy learning:

.. math::

    \max_{\theta} \underset{s \sim {\mathcal D}}{{\mathrm E}}\left[ Q_{\phi}(s,\mathbf{a}, \mu_{\theta}(s)) \right]

Here :math:`{\mathcal D}` is the replay buffer, which can be shared across agents.
:math:`\mathbf{a}` is an action set, including opponents.
:math:`r` is the reward.
:math:`\mathbf{s}` is the observation/state set, including opponents.
:math:`\mathbf{s'}` is the next observation/state set, including opponents.
:math:`d` is set to 1(True) when an episode ends else 0(False).
:math:`{\gamma}` is discount value.
:math:`\mu_{\theta}` is policy net, which can be shared across agents.
:math:`Q_{\phi}` is Q net, which can be shared across agents.
:math:`g_{\psi}` is mixing network.
:math:`\mu_{\theta_{\text{targ}}}` is target policy net, which can be shared across agents.
:math:`Q_{\phi_{\text{targ}}}` is target Q net, which can be shared across agents.
:math:`g_{\psi_{\text{targ}}}` is target mixing network.



Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We extend the vanilla DDPG of RLlib to be recurrent neural network(RNN) compatible.
Based on RNN compatible DDPG, we add the centralized sampling and training module to the original pipeline.
The main differences between IDDPG and MADDPG are:

- model side: the agent model-related modules and functions are built in a value decomposition style:
    - ``build_facmac_models_and_action_dist``
    - ``FACMAC_RNN_TorchModel``
- algorithm side: the sampling and training pipelines are built in a value decomposition style:
    - ``q_value_mixing``
    - ``value_mixing_ddpg_loss``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/maddpg``
- ``marl/algos/hyperparams/fintuned/env/maddpg``


.. admonition:: You Should Know

    - The policy inference procedure of FACMAC is kept the same as IDDPG.
    - Some tricks like `gumble softmax` enables FACMAC net to output categorical-like action distribution.

---------------------

Read List
-------------

- `Continuous Control with Deep Reinforcement Learning <https://arxiv.org/abs/1509.02971>`_
- `Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments <https://arxiv.org/abs/1706.02275>`_
- `FACMAC: Factored Multi-Agent Centralised Policy Gradients <https://arxiv.org/pdf/2003.06709.pdf>`_
