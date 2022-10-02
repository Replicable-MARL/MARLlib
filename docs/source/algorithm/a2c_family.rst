Advanced Actor Critic Family
======================================================================

.. contents::
    :local:
    :depth: 3

---------------------

.. _A2C:

Advanced Actor-Critic: A Recap
-----------------------------------------------

**Preliminary**:

- Vanilla Policy Gradient (PG)
- Monte Carlo Policy Gradients (REINFORCE)

Why do we need an advanced actor-critic (A2C)? Before A2C, we already have some policy gradient method variants like REINFORCE. However, these methods are not stable in training. This is due to
the large variance in the reward signal, which is used to update the policy. A solution to reduce this variance is introducing a baseline for it. A2C adopts a critic value function conditioned on **state**
as the baseline and compute the difference between the state value and Q value as the **advantage**.

.. math::

    A(s_t,a_t) = Q(s_t,a_t) - V(s_t)

Now we need two functions :math:`Q` and :math:`V` to estimate :math:`A`. Luckily we can do some transformations for the above equation.
Recall the bellman optimality equation:

.. math::

    Q(s_t,a_t)  = r_{t+1} + \lambda V(s_{t+1})

:math:`A` can be written as:

.. math::

    A_t = r_{t+1} + \lambda V(s_{t+1}) - V(s_t)

In this way, only :math:`V` is needed to estimate :math:`A`
Finally we use policy gradient to update the :math:`V` function by:

.. math::

    \nabla_\theta J(\theta) \sim \sum_{t=0}^{T-1}\nabla_\theta \log\pi_{\theta}(a_t|s_t)A_t

The only thing left is how to update the parameters of the critic function:

.. math::

    \phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( V_{\phi} (s_t) - \hat{R}_t \right)^2


Here
:math:`V` is the critic function.
:math:`\phi` is the parameters of the critic function.
:math:`{\mathcal D}` is the collected trajectories.
:math:`R` is the rewards-to-go.
:math:`\tau` is the trajectory.


---------------------

.. _IA2C:

IA2C: multi-agent version of A2C
-----------------------------------------------------

.. admonition:: Quick Facts

    - Independent advanced actor-critic (IA2C) is a natural extension of standard advanced actor-critic (A2C) in multi-agent settings.
    - Agent architecture of IA2C consists of two modules: ``policy`` and ``critic``.
    - IA2C is applicable for cooperative, collaborative, competitive, and mixed task modes.

**Preliminary**:

- :ref:`A2C`

Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In IA2C, each agent follows a standard A2C sampling/training pipeline. Therefore, IA2C is a general baseline for all MARL tasks with robust performance.
Note that buffer and agent models can be shared or separately trained across agents. And this applies to all algorithms in the A2C family.

.. figure:: ../images/ia2c.png
    :width: 600
    :align: center

    Independent Advanced Actor-Critic (IA2C)

Characteristic
^^^^^^^^^^^^^^^

action space

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``discrete``
     - ``continuous``

task mode

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 0

   * - ``cooperative``
     - ``collaborative``
     - ``competitive``
     - ``mixed``

taxonomy label

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``on-policy``
     - ``stochastic``
     - ``independent learning``


Insights
^^^^^^^^^^^^^^^^^^^^^^^


IA2C is the simplest multi-agent version of standard A2C. Each agent is now an A2C-based sampler and learner.
IA2C does not need information sharing.
While knowledge sharing across agents is optional in IA2C.

.. admonition:: Information Sharing

    In multi-agent learning, the concept of information sharing is not well defined and may confuse.
    Here we try to clarify this by categorizing the type of information sharing into three.

    - real/sampled data: observation, action, etc.
    - predicted data: Q/critic value, message for communication, etc.
    - knowledge: experience replay buffer, model parameters, etc.

    Knowledge-level information sharing is usually excluded from information sharing and is only seen as a trick.
    However, recent works find it is essential for good performance. Here, we include knowledge sharing as part of the information sharing.

Mathematical Form
^^^^^^^^^^^^^^^^^^

Standing at the view of a single agent, the mathematical formulation of IA2C is similiar as :ref:`A2C`, except that in MARL,
agent usually has no access to the global state typically under partial observable setting. Therefore, we use :math:`o` for
local observation and :math:`s`for the global state. We then rewrite the mathematical formulation of A2C as:

Critic learning: every iteration gives a better value function.

.. math::

    \phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( V_{\phi} (o_t) - \hat{R}_t \right)^2

Advantage Estimation: how good are current action regarding to the baseline critic value.

.. math::

    A_t = r_{t+1} + \lambda V_{\phi} (o_{t+1}) - V_{\phi} (o_t)

Policy learning: computing the policy gradient using estimated advantage to update the policy function.

.. math::

    \nabla_\theta J(\theta) \sim \sum_{t=0}^{T-1}\nabla_\theta \log\pi_{\theta}(u_t|o_t)A_t



Note that in multi-agent settings, all the agent models can be shared, including:

- value function :math:`V_{\phi}`.
- policy function :math:`\pi_{\theta}`.


Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We use vanilla A2C implementation of RLlib in IA2C.

Key hyperparameter location:

- ``marl/algos/hyperparams/common/a2c``
- ``marl/algos/hyperparams/fintuned/env/a2c``



---------------------

.. _MAA2C:

MAA2C: A2C agent with a centralized critic
-----------------------------------------------------

.. admonition:: Quick Facts

    - Multi-agent advanced actor-critic (MAA2C) is one of the extended versions of :ref:`IA2C`.
    - Agent architecture of MAA2C consists of two models: ``policy`` and ``critic``.
    - MAA2C is applicable for cooperative, collaborative, competitive, and mixed task modes.

**Preliminary**:

- :ref:`IA2C`

Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the sampling stage, agents share information with others. The information includes others' observations and predicted actions. After collecting the necessary information from other agents,
all agents follow the standard A2C training pipeline, except using the centralized critic value function to calculate the GAE and conduct the A2C critic learning procedure.

.. figure:: ../images/maa2c.png
    :width: 600
    :align: center

    Multi-agent Advanced Actor-Critic (MAA2C)


Characteristic
^^^^^^^^^^^^^^^

action space

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``discrete``
     - ``continuous``

task mode

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 0

   * - ``cooperative``
     - ``collaborative``
     - ``competitive``
     - ``mixed``

taxonomy label

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``on-policy``
     - ``stochastic``
     - ``centralized critic``



Insights
^^^^^^^^^^^^^^^^^^^^^^^

Centralized critic enables MAPPO to gain a strong performance in MARL. The same architecture can also be applied to IA2C.
In practice, MAA2C can also perform well in most scenarios.
There is no official MAA2C paper, and we implement MAA2C in the same pipeline as MAPPO but with an advanced actor-critic loss.


Mathematical Form
^^^^^^^^^^^^^^^^^^

MAA2C needs information sharing across agents. Critic learning utilizes self-observation and global information,
including state and actions. Here we bold the symbol (e.g., :math:`u` to :math:`\mathbf{u}`) to indicate that more than one agent information is contained.

Critic learning: every iteration gives a better value function.

.. math::

    \phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( V_{\phi} (o_t,s_t,\mathbf{u_t^-}) - \hat{R}_t \right)^2

Advantage Estimation: how good are current action regarding to the baseline critic value.

.. math::

    A_t = r_{t+1} + \lambda V_{\phi} (o_{t+1},s_{t+1},\mathbf{u_{t+1}^-}) - V_{\phi} (o_t,s_t,\mathbf{u_t^-})

Policy learning: computing the policy gradient using estimated advantage to update the policy function.

.. math::

    \nabla_\theta J(\theta) \sim \sum_{t=0}^{T-1}\nabla_\theta \log\pi_{\theta}(u_t|o_t)A_t

Here
:math:`\mathcal D` is the collected trajectories that can be shared across agents.
:math:`R` is the rewards-to-go.
:math:`\tau` is the trajectory.
:math:`A` is the advantage.
:math:`\gamma` is discount value.
:math:`\lambda` is the weight value of GAE.
:math:`o` is the current agent local observation.
:math:`u` is the current agent action.
:math:`\mathbf{u}^-` is the action set of all agents, except the current agent.
:math:`s` is the current agent global state.
:math:`V_{\phi}` is the critic value function, which can be shared across agents.
:math:`\pi_{\theta}` is the policy function, which can be shared across agents.

Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

Based on IA2C, we add centralized modules to implement MAA2C.
The details can be found in:

- ``centralized_critic_postprocessing``
- ``central_critic_a2c_loss``
- ``CC_RNN``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/maa2c``
- ``marl/algos/hyperparams/fintuned/env/maa2c``

---------------------

.. _COMA:

COMA: MAA2C with Counterfactual Multi-Agent Policy Gradients
-----------------------------------------------------

.. admonition:: Quick Facts

    - Counterfactual multi-agent policy gradients (COMA) is based on MAA2C.
    - Agent architecture of COMA consists of two models: ``policy`` and ``Q``.
    - COMA adopts a counterfactual baseline to marginalize a single agent’s action's contribution.
    - COMA is applicable for cooperative, collaborative, competitive, and mixed task modes.

**Preliminary**:

- :ref:`IA2C`
- :ref:`MAA2C`

Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the sampling stage, agents share information with others. The information includes others' observations and predicted actions. After collecting the necessary information from other agents,
all agents follow the A2C training pipeline but use COMA loss to update the policy. The value function (critic) is centralized the same as MAA2C.

.. figure:: ../images/coma.png
    :width: 600
    :align: center

    Counterfactual Multi-Agent Policy Gradients (COMA)


Characteristic
^^^^^^^^^^^^^^^

action space

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``discrete``

task mode

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 0

   * - ``cooperative``
     - ``collaborative``
     - ``competitive``
     - ``mixed``

taxonomy label

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``on-policy``
     - ``stochastic``
     - ``centralized critic``



Insights
^^^^^^^^^^^^^^^^^^^^^^^

Efficiently learning decentralized policies is an essential demand for modern AI systems. However, assigning credit to an agent becomes a significant challenge when only one global reward exists.
COMA provides one solution for this problem:

#. COMA uses a counterfactual baseline that marginalizes a single agent’s action while keeping the other agents’ actions fixed.
#. COMA develops a centralized Q that allows the counterfactual baseline to be computed efficiently in a single forward pass.
#. COMA significantly improves average performance over other multi-agent actor-critic methods under decentralized execution and partial observability settings.

.. admonition:: You Should Know

    - Although COMA is based on stochastic policy gradient methods, it is only evaluated in discrete action space. Extending to continuous action space requires additional tricks on computing critic value (which is not good news for stochastic methods)
    - In recent years' research, COMA's has been proven to be relatively worse in solving tasks like :ref:`SMAC` and :ref:`MPE` than other on-policy methods, even basic independent methods like :ref:`IA2C`.

Mathematical Form
^^^^^^^^^^^^^^^^^^

COMA needs information sharing across agents. Q learning utilizes self-observation and global information,
including state and actions. The advantage estimation is based on counterfactual baseline, which is different from other algorithms in A2C family.

Q learning: every iteration gives a better Q function.

.. math::

    \phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( Q_{\phi} (o_t, s_t, u_t, (\mathbf{u_t}^-)) - \hat{R}_t \right)^2

Marginalized Advantage Estimation: how good are current action's Q value compared to the average Q value of the whole action space.

.. math::

    A_t = Q_{\phi}(o_t, s_t, u_t, \mathbf{a}^-) - \sum_{u_t} \pi(u_t \vert \tau) Q_{\phi}(o_t, s_t, u_t, (\mathbf{u_t}^-))


Policy learning:

.. math::

    L(o, s, a, \mathbf{a}^-, \theta)=\log\pi_\theta(a|s)A((o, s, a, \mathbf{a}^-)

Here
:math:`{\mathcal D}` is the collected trajectories.
:math:`R` is the rewards-to-go.
:math:`\tau` is the trajectory.
:math:`A` is the advantage.
:math:`o` is the current agent local observation.
:math:`u` is the current agent action.
:math:`\mathbf{u}^-` is the action set of all agents, except the current agent.
:math:`s` is the global state.
:math:`Q_{\phi}` is the Q function.
:math:`\pi_{\theta}` is the policy function.

Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

Based on IA2C, we add the COMA loss function.
The details can be found in:

- ``centralized_critic_postprocessing``
- ``central_critic_coma_loss``
- ``CC_RNN``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/coma``
- ``marl/algos/hyperparams/fintuned/env/coma``

---------------------

.. _VDA2C:


VDA2C: mixing a bunch of A2C agents' critics
-----------------------------------------------------

.. admonition:: Quick Facts

    - Value decomposition advanced actor-critic (VDA2C) is one of the extensions of :ref:`IA2C`.
    - Agent architecture of VDA2C consists of three modules: ``policy``, ``critic``, and ``mixer``.
    - VDA2C is proposed to solve cooperative and collaborative tasks only.

**Preliminary**:

- :ref:`IA2C`
- :ref:`QMIX`

Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the sampling stage, agents share information with others. The information includes others' observations and predicted critic value. After collecting the necessary information from other agents,
all agents follow the standard A2C training pipeline, except for using the mixed critic value to calculate the GAE and conduct the A2C critic learning procedure.

.. figure:: ../images/vda2c.png
    :width: 600
    :align: center

    Value Decomposition Advanced Actor-Critic (VDA2C)

Characteristic
^^^^^^^^^^^^^^^

action space

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``discrete``
     - ``continuous``

task mode

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``cooperative``
     - ``collaborative``



taxonomy label

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``on-policy``
     - ``stochastic``
     - ``value decomposition``



Insights
^^^^^^^^^^^^^^^^^^^^^^^

VDA2C focuses on credit assignment learning, similar to the joint Q learning family. However, compared to the joint Q learning family, VDA2C adopts on-policy learning and mixes the V function instead of the Q function.
The sampling efficiency of VDA2C is worse than joint Q learning algorithms. VDA2C is applicable for both discrete and continuous control problems.

Mathematical Form
^^^^^^^^^^^^^^^^^^

VDA2C needs information sharing across agents. Therefore, the critic mixing utilizes both self-observation and other agents' observation.
Here we bold the symbol (e.g., :math:`u` to :math:`\mathbf{u}`) to indicate that more than one agent information is contained.


Critic mixing:

.. math::

    V_{tot}(\mathbf{u}, s;\boldsymbol{\phi},\psi) = g_{\psi}\bigl(s, V_{\phi_1},V_{\phi_2},..,V_{\phi_n} \bigr)


Mixed Critic learning: every iteration gives a better value function and a better mixing function.


.. math::

    \phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( V_{tot} - \hat{R}_t \right)^2

Advantage Estimation: how good are current joint action set regarding to the baseline critic value.

.. math::

    A_t = r_{t+1} + \lambda V_{tot}^{t+1} - V_{tot}^{t}

Policy learning: computing the policy gradient using estimated advantage to update the policy function.

.. math::

    \nabla_\theta J(\theta) \sim \sum_{t=0}^{T-1}\nabla_\theta \log\pi_{\theta}(u_t|s_t)A_t

Here
:math:`\mathcal D` is the collected trajectories that can be shared across agents.
:math:`R` is the rewards-to-go.
:math:`\tau` is the trajectory.
:math:`A` is the advantage.
:math:`\gamma` is discount value.
:math:`\lambda` is the weight value of GAE.
:math:`o` is the current agent local observation.
:math:`u` is the current agent action.
:math:`\mathbf{u}^-` is the action set of all agents, except the current agent.
:math:`s` is the current agent global state.
:math:`V_{\phi}` is the critic value function, which can be shared across agents.
:math:`\pi_{\theta}` is the policy function, which can be shared across agents.
:math:`g_{\psi}` is a mixing network, which must be shared across agents.



Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

Based on IA2C, we add mixing Q modules to implement VDA2C.
The details can be found in:

- ``value_mixing_postprocessing``
- ``value_mix_actor_critic_loss``
- ``VD_RNN``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/vda2c``
- ``marl/algos/hyperparams/fintuned/env/vda2c``


---------------------


Read List
-------------

- `Advanced Actor-Critic Algorithms <https://arxiv.org/abs/1707.06347>`_
- `The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games <https://arxiv.org/abs/2103.01955>`_
- `Counterfactual Multi-Agent Policy Gradients <https://ojs.aaai.org/index.php/AAAI/article/download/11794/11653>`_
- `Value-Decomposition Multi-Agent Actor-Critics <https://arxiv.org/abs/2007.12306>`_
