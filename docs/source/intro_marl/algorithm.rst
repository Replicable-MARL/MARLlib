.. _algorithm-detail:

*******************************************
Part 3: MARL Baseline Algorithms
*******************************************

Algorithm list of MARLlib, including the mathematical formulation and ``MARLlib`` style of implementation.

.. contents::
    :local:
    :depth: 2

Etiam turis ante, luctus sed velit tristique, finibus volutpat dui. Nam sagittis vel ante nec malesuada.
Praesent dignissim mi nec ornare elementum. Nunc eu augue vel sem dignissim cursus sed et nulla.
Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas.
Pellentesque dictum dui sem, non placerat tortor rhoncus in. Sed placerat nulla at rhoncus iaculis.

Independent Learning
========================

Features of independent learning


.. _IQL:

Independent Q Learning (IQL)
---------------------------------------------



Supported action space:

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``discrete``

Supported task mode:

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``cooperative``
     - ``collaborative``
     - ``competitive``


Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam efficitur in eros et blandit. Nunc maximus,

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `Human-level control through deep reinforcement learning <https://daiwk.github.io/assets/dqn.pdf>`_
- `Deep Recurrent Q-learning for Partially Observable MDPs <https://www.aaai.org/ocs/index.php/FSS/FSS15/paper/download/11673/11503>`_



.. _IPG:

Independent Policy Gradient (IPG)
---------------------------------------------


Supported action space:

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``discrete``
     - ``continues``

Supported task mode:

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``cooperative``
     - ``collaborative``
     - ``competitive``

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam efficitur in eros et blandit. Nunc maximus,

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `Policy gradient methods for reinforcement learning with function approximation <https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf>`_


.. _IA2C:

Independent Advanced Actor Critic (IA2C)
---------------------------------------------


Supported action space:

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``discrete``
     - ``continues``

Supported task mode:

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``cooperative``
     - ``collaborative``
     - ``competitive``

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam efficitur in eros et blandit. Nunc maximus,

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `Asynchronous Methods for Deep Reinforcement Learning <https://arxiv.org/abs/1602.01783>`_


.. _IDDPG:

Independent Deep Deterministic Policy Gradient (IDDPG)
-------------------------------------------------------------

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

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Q-Learning & Deep Q Network(DQN)

Algorithm Description
^^^^^^^^^^^^^^^^^^^^^^^

Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.
The motivation of DDPG is to tackling the problem that standard Q-learning can only be used in discrete action space (a finite number of actions).
To extend Q function to continues control problem, DDPG adopts an extra policy network :math:`\mu(s)` parameterized by :math:`\theta` to produce action value.
Then Q value is calculated as :math:`Q(s,\mu(s))` and the Q network is parameterized by :math:`\phi`.

Math Formulation
^^^^^^^^^^^^^^^^^^

Q learning:

.. math::

    L(\phi, {\mathcal D}) = \underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[
        \Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) Q_{\phi_{\text{targ}}}(s', \mu_{\theta_{\text{targ}}}(s')) \right) \Bigg)^2
        \right],

Policy learning:

.. math::

    \max_{\theta} \underset{s \sim {\mathcal D}}{{\mathrm E}}\left[ Q_{\phi}(s, \mu_{\theta}(s)) \right].

Here :math:`{\mathcal D}` is the replay buffer, which can be shared across agents.
:math:`a` is the action taken.
:math:`s` is the observation/state.
:math:`s'` is the next observation/state.
:math:`d` is 1 when episode ends else 0.
:math:`\gamma` is discount value.
:math:`\mu_{\theta_{\text{targ}}}` is policy target net, which can be shared across agents.
:math:`\phi_{\text{targ}}` is Q target net, which can be shared across agents.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each agent follows the standard DDPG learning pipeline as described in Preliminary. No information is shared across agents.

.. figure:: ../images/IDDPG.png
    :width: 400
    :align: center

    Independent Deep Deterministic Policy Gradient (IDDPG)

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `Continuous control with deep reinforcement learning <https://arxiv.org/abs/1509.02971>`_



.. _ITRPO:

Independent Trust Region Policy Optimization (ITRPO)
-------------------------------------------------------------


Supported action space:

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``discrete``
     - ``continues``

Supported task mode:

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``cooperative``
     - ``collaborative``
     - ``competitive``

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam efficitur in eros et blandit. Nunc maximus,

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `Trust Region Policy Optimization <http://proceedings.mlr.press/v37/schulman15.pdf>`_

--------------

.. _IPPO:

Independent Proximal Policy Optimization (IPPO)
-----------------------------------------------------

Supported action space:

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``discrete``
     - ``continues``

Supported task mode:

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``cooperative``
     - ``collaborative``
     - ``competitive``

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam efficitur in eros et blandit. Nunc maximus,

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_
- `Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge? <https://arxiv.org/abs/2011.09533>`_



Centralized Critic
========================

Features of centralized critic under CTDE framework.

.. _MADDPG:

Multi-agent Deep Deterministic Policy Gradient (MADDPG)
-------------------------------------------------------------

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

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Deep Deterministic Policy Gradient(DDPG).

Algorithm Description
^^^^^^^^^^^^^^^^^^^^^^^

Multi-agent Deep Deterministic Policy Gradient (MADDPG) is an algorithm extends DDPG with a centralied Q function that takes not only observation and action from current agent,
but also other agents. Similiar to DDPG, MADDPG also has a policy network :math:`\mu(s)` parameterized by :math:`\theta` to produce action value.
While the Centralized Q value is calculated as :math:`Q(s,\mu(s))` and the Q network is parameterized by :math:`\phi`.

which concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.
The motivation of DDPG is to tackling the problem that standard Q-learning can only be used in discrete action space (a finite number of actions).
To extend Q function to continues control problem, DDPG adopts an extra policy network :math:`\mu(s)` parameterized by :math:`\theta` to produce action value.
Then Q value is calculated as :math:`Q(s,\mu(s))` and the Q network is parameterized by :math:`\phi`.

Math Formulation
^^^^^^^^^^^^^^^^^^

Q learning:

.. math::

    L(\phi, {\mathcal D}) = \underset{(\mathbf{s},\mathbf{a},r,\mathbf{s'},d) \sim {\mathcal D}}{{\mathrm E}}\left[
        \Bigg( Q_{\phi}(\mathbf{s},\mathbf{a}) - \left(r + \gamma (1 - d) Q_{\phi_{\text{targ}}}(\mathbf{s'}, \mu_{\theta_{\text{targ}}}(\mathbf{s'})) \right) \Bigg)^2
        \right],


Policy learning:

.. math::

    \max_{\theta} \underset{s \sim {\mathcal D}}{{\mathrm E}}\left[ Q_{\phi}(s, \mu_{\theta}(s)) \right].

Here :math:`{\mathcal D}` is the replay buffer, which can be shared across agents.
:math:`\mathbf{a}` is the action taken, including opponents.
:math:`\mathbf{s}` is the observation/state, including opponents.
:math:`\mathbf{s'}` is the next observation/state, including opponents.
:math:`d` is 1 when episode ends else 0.
:math:`\gamma` is discount value.
:math:`\mu_{\theta_{\text{targ}}}` is policy target net, which can be shared across agents.
:math:`\phi_{\text{targ}}` is Q target net, which can be shared across agents.

Note: Policy learning procedure of MADDPG is exactly same as DDPG/IDDPG


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each agent follows the standard DDPG learning pipeline as described in Preliminary. No information is shared across agents.

.. figure:: ../images/MADDPG.png
    :width: 400
    :align: center

    Multi-agent Deep Deterministic Policy Gradient (MADDPG)

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments <https://arxiv.org/abs/1706.02275>`_

.. _COMA:

Counterfactual Multi-Agent Policy Gradients (COMA)
-----------------------------------------------------


Supported action space:

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``discrete``

Supported task mode:

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``cooperative``
     - ``collaborative``
     - ``competitive``

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam efficitur in eros et blandit. Nunc maximus,

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `Counterfactual Multi-Agent Policy Gradients <https://ojs.aaai.org/index.php/AAAI/article/download/11794/11653>`_


.. _MAA2C:

Multi-agent Advanced Actor Critic (MAA2C)
---------------------------------------------

Supported action space:

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``discrete``
     - ``continues``

Supported task mode:

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``cooperative``
     - ``collaborative``
     - ``competitive``

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam efficitur in eros et blandit. Nunc maximus,

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^





.. _MATRPO:

Multi-agent Trust Region Policy Optimization (MATRPO)
-------------------------------------------------------------

Supported action space:

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``discrete``
     - ``continues``

Supported task mode:

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``cooperative``
     - ``collaborative``
     - ``competitive``

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam efficitur in eros et blandit. Nunc maximus,

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. _MAPPO:

Multi-agent Proximal Policy Optimization (MAPPO)
-----------------------------------------------------


Supported action space:

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``discrete``
     - ``continues``

Supported task mode:

.. list-table::
   :widths: 25 25 25
   :header-rows: 0

   * - ``cooperative``
     - ``collaborative``
     - ``competitive``

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam efficitur in eros et blandit. Nunc maximus,

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games <https://arxiv.org/abs/2103.01955>`_


.. _HATRPO:

Heterogeneous Multi-agent Trust Region Policy Optimization (HATRPO)
------------------------------------------------------------------------


Supported action space:

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``discrete``
     - ``continues``

Supported task mode:

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``cooperative``

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam efficitur in eros et blandit. Nunc maximus,

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning <https://arxiv.org/abs/2109.11251>`_


.. _HAPPO:

Heterogeneous Multi-agent Proximal Policy Optimization (HAPPO)
----------------------------------------------------------------

Supported action space:

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``discrete``
     - ``continues``

Supported task mode:

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``cooperative``

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam efficitur in eros et blandit. Nunc maximus,

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Value Decomposition
========================

Features of value decomposition under CTDE framework.

.. _VDN:

Value Decomposition Networks (VDN)
---------------------------------------------


Supported action space:

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``discrete``

Supported task mode:

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``cooperative``

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam efficitur in eros et blandit. Nunc maximus,

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `Value-Decomposition Networks For Cooperative Multi-Agent Learning <https://arxiv.org/abs/1706.05296>`_

.. _QMIX:

Monotonic Value Function Factorisation (QMIX)
---------------------------------------------


Supported action space:

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``discrete``

Supported task mode:

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``cooperative``

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam efficitur in eros et blandit. Nunc maximus,

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


- `QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning <https://arxiv.org/abs/1803.11485>`_


.. _FACMAC:

Factored Multi-Agent Centralised Policy Gradients (FACMAC)
-------------------------------------------------------------



Supported action space:

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``continues``

Supported task mode:

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``cooperative``

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam efficitur in eros et blandit. Nunc maximus,

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `FACMAC: Factored Multi-Agent Centralised Policy Gradients <https://arxiv.org/abs/2003.06709>`_



.. _VDA2C:

Value Decomposition Advanced Actor Critic (VDA2C)
-------------------------------------------------------



Supported action space:

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``discrete``
     - ``continues``

Supported task mode:

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``cooperative``

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam efficitur in eros et blandit. Nunc maximus,

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `Value-Decomposition Multi-Agent Actor-Critics <https://arxiv.org/abs/2007.12306>`_

.. _VDPPO:

Value Decomposition Proximal Policy Optimization (VDPPO)
-------------------------------------------------------------

Supported action space:

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - ``discrete``
     - ``continues``

Supported task mode:

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``cooperative``

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam efficitur in eros et blandit. Nunc maximus,

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Donec non rutrum lorem. Aenean sagittis metus at pharetra fringilla. Nunc sapien dolor, cursus sed nisi at,
pretium tristique lectus. Sed pellentesque leo lectus, et convallis ipsum euismod a.

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
