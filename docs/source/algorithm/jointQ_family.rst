Joint Q Learning Family
======================================================================

.. contents::
    :local:
    :depth: 3

---------------------

.. _DQN:

Deep (Recurrent) Q Learning: A Recap
-----------------------------------------------

**Vanilla Q Learning**

Without deep learning to approximate the state-action pair value, Q learning is restricted in the Q-table, which records all the pairs of :math:`(s,a)` with their corresponding Q values.
The Q-Learning algorithm aims to learn the Q-Value of each :math:`(s,a)` pair based on interaction with the environment and iteratively update the Q-value record in the Q-table.

The first step is choosing the action using **Epsilon-Greedy Exploration Strategy**.
The agent then takes the best-known action or does random exploration, entering the next state with a reward given by the environment.

The second step uses the bellman equation to update the Q-table based on collected data.

.. math::

    Q(s,a)=(1-\alpha)Q(s,a)+\alpha*(r+\lambda*max_a(s^{'},a^{'}))

Here
:math:`s` is the state.
:math:`a` is the action.
:math:`s^{'}` is the next state.
:math:`a^{'}` is the next action that yields the highest Q value
:math:`\alpha` is the learning rate.
:math:`\lambda` is the discount factor.

Keeping iterating these two steps and updating the Q-table can converge the Q value. And the final Q value is the reward expectation of the action you choose based on the current state.

**Deep Q Learning**

Introducing deep learning into Q learning is a giant leap. However, it enables us to approximate the Q value using a neural network.
There are two networks in deep Q learning:math:`Q` network and math:`Q_{tag}` network, which are the same in architecture.

Q table is now replaced by :math:`Q` network to approximate the Q value.
This way, the :math:`(s,a)` pairs number can be huge. As we can encode the state to a feature vector and learn the mapping between
the feature vector and the Q value.
The design of the Q function is simple, it takes :math:`s` as input and has :math:`a` number of output dimensions.
The max value of output nodes is chosen to be :math:`max_a(s^{'},a^{'})` in vanilla Q learning.
Finally, we use the Bellman equation to update the network.
The optimization target is to minimize the minimum square error (MSE) between the current Q value estimation and the target Q estimation.

.. math::

    \phi_{k+1} = \arg \min_{\phi}(Q_\phi(s,a)-(r+\lambda*max_{a^{'}}Q_{\phi_{tar}}(s^{'},a^{'})))^2

The :math:`Q_{tag}` network is updated every :math:`t` timesteps copying the :math:`Q` network.

**DQN + Recurrent Neural Network(DRQN)**

When we do not have full access to the state information, we need to record the history (trajectory information) to help the action chosen.
RNN is then introduced to deep Q learning to deal with the partial observable Markov Decision Process(POMDP) by encoding the history into the hidden state.
The optimization target now becomes:

.. math::

    \phi_{k+1} = \arg \min_{\phi}(Q_\phi(o,h,a)-(r+\lambda*max_{a^{'}}Q_{\phi_{tar}}(o^{'},h^{'},a^{'})))^2

Here
:math:`o` is the observation as we cannot access the state :math:`s`.
:math:`o^{'}` is the next observation.
:math:`h` is the hidden state(s) of the RNN.
:math:`h^{'}` is the next hidden state(s) of the RNN.

.. admonition:: You Should Know

    Navigating from DQN to DRQN, you need to:

    - replace the deep Q net's multi-layer perceptron(MLP) module with a recurrent module, e.g., GRU, LSTM.
    - store the data in episode format. (while DQN has no such restriction)

---------------------

.. _IQL:

IQL: multi-agent version of D(R)QN.
-----------------------------------------------------

.. admonition:: Quick Facts

    - Independent Q Learning (IQL) is the natural extension of q learning under multi-agent settings.
    - Agent architecture of IQL consists of one module: ``Q``.
    - IQL is applicable for cooperative, collaborative, competitive, and mixed task modes.

Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In IQL, each agent follows a standard D(R)QN sampling/training pipeline.

.. figure:: ../images/iql.png
    :width: 600
    :align: center

    Independent Q Learning (IQL)

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

   * - ``off-policy``
     - ``stochastic``
     - ``independent learning``


Insights
^^^^^^^^^^^^^^^^^^^^^^^

**Preliminary**

- :ref:`DQN`

IQL treats each agent in a multi-agent system as a single agent and uses its own collected data as input to conduct the standard DQN or DRQN learning procedure.
No information sharing is needed.
While knowledge sharing across agents is optional in IQL.

.. admonition:: Information Sharing

    In multi-agent learning, the concept of information sharing is not well defined and may confuse.
    Here we try to clarify this by categorizing the type of information sharing into three.

    - real/sampled data: observation, action, etc.
    - predicted data: Q/critic value, message for communication, etc.
    - knowledge: experience replay buffer, model parameters, etc.

    Knowledge-level information sharing is usually excluded from information sharing and is only seen as a trick.
    However, recent works find it is essential for good performance. Here, we include knowledge sharing as part of the information sharing.


Math Formulation
^^^^^^^^^^^^^^^^^^

Standing at the view of a single agent, the mathematical formulation of IQL is the same as :ref:`DQN`.

Note in multi-agent settings, all the agent models and buffer can be shared, including:

- replay buffer :math:`{\mathcal D}`.
- Q function :math:`Q_{\phi}`.
- target Q function :math:`Q_{\phi_{\text{targ}}}`.



Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We use vanilla IQL implementation of RLlib, but with further improvement to ensure the performance is aligned with the official implementation.
The differences between ours and vanilla IQL can be found in

- ``episode_execution_plan``
- ``EpisodeBasedReplayBuffer``
- ``JointQLoss``
- ``JointQPolicy``

Key hyperparameters location:

- ``marl/algos/hyperparams/common/iql``
- ``marl/algos/hyperparams/finetuned/env/iql``

---------------------

.. _VDN:


VDN: mixing Q with value decomposition network
-----------------------------------------------------

.. admonition:: Quick Facts

    - Value Decomposition Network(VDN) is one of the value decomposition versions of IQL.
    - Agent architecture of VDN consists of one module: ``Q`` network.
    - VDN is applicable for cooperative and collaborative task modes.

Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In VDN, each agent follows a standard D(R)QN sampling pipeline. And sharing its Q value and target Q value with other agents before entering the training loop.
In the training loop, the Q value and target Q value of the current agent and other agents are summed to get the :math:`Q_{tot}`.


.. figure:: ../images/vdn.png
    :width: 600
    :align: center

    Value Decomposition Network (VDN)


Characteristic
^^^^^^^^^^^^^^^

action space

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``discrete``


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

   * - ``off-policy``
     - ``stochastic``
     - ``value decomposition``


Algorithm Insights
^^^^^^^^^^^^^^^^^^^^^^^

Preliminary

- :ref:`IQL`

Optimizing multiple agents' joint policy with a single team reward can be very challenging as the action and observation space is now too large when combined.
Value Decomposition Network(VDN) is the first proposed algorithm for this problem. The solution is relatively straightforward:

- Each agent is still a standard ``Q``, use self-observation as input and output the action logits(Q value).
- The Q values of all agents are added together for mixed Q value annotated as :math:`Q_{tot}`
- Using standard DQN to optimize the Q net using :math:`Q_{tot}` with the team reward :math:`r`.
- The gradient each Q net received depends on the **contribution** of its Q value to the :math:`Q_{tot}`:
The Q net that outputs a larger Q will be updated more; the smaller will be updated less.

The value decomposition version of IQL is also referred as **joint Q learning**(JointQ).
These two names emphasize different aspects. Value decomposition focuses on how the team reward is divided to update the Q net, known as credit assignment.
Joint Q learning shows how the optimization target :math:`Q_{tot}` is got.
As VDN is developed to address the cooperative multi-agent task, sharing the parameter is the primary option, which brings higher data efficiency and a smaller model size.

.. admonition:: You Should Know:

    VDN is the first value decomposition algorithm for cooperative multi-agent tasks. However, simply summing the Q value can reduce the diversity of
    the policy and can quickly stuck into local optimum, especially when the Q net is shared across agents.


Math Formulation
^^^^^^^^^^^^^^^^^^

VDN needs information sharing across agents. Here we bold the symbol (e.g., :math:`o` to :math:`\mathbf{o}`) to indicate that more than one agent information is contained.


Q sum: add all the Q values to get the total Q value

.. math::

    Q_{\phi}^{tot} = \sum_{i=1}^{n} Q_{\phi}^i

Q learning: every iteration get a better total Q value estimation, passing gradient to each Q function to update it.

.. math::

    L(\phi, {\mathcal D}) = \underset{\tau \sim {\mathcal D}}{{\mathrm E}}\Bigg(Q_{\phi}^{tot} - \left(r + \gamma (1 - d) Q_{\phi_{targ}}^{tot^{'}} \right) \Bigg)^2


Here :math:`{\mathcal D}` is the replay buffer, which can be shared across agents.
:math:`r` is the reward.
:math:`d` is set to 1(True) when an episode ends else 0(False).
:math:`{\gamma}` is discount value.
:math:`Q_{\phi}` is Q net, which can be shared across agents.
:math:`Q_{\phi_{\text{targ}}}` is target Q net, which can be shared across agents.

Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We use vanilla VDN implementation of RLlib, but with further improvement to ensure the performance is aligned with the official implementation.
The differences between ours and vanilla VDN can be found in

- ``episode_execution_plan``
- ``EpisodeBasedReplayBuffer``
- ``JointQLoss``
- ``JointQPolicy``

Key hyperparameters location:

- ``marl/algos/hyperparams/common/vdn``
- ``marl/algos/hyperparams/finetuned/env/vdn``


----------------

.. _QMIX:

QMIX: mixing Q with monotonic factorization
-----------------------------------------------------------------


.. admonition:: Quick Facts

    - Monotonic Value Function Factorisation(QMIX) is one of the value decomposition versions of IQL.
    - Agent architecture of QMIX consists of two modules: ``Q`` and ``Mixer``.
    - QMIX is applicable for cooperative and collaborative task modes.

Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In QMIX, each agent follows a standard D(R)QN sampling pipeline. And sharing its Q value and target Q value with other agents before entering the training loop.
In the training loop, the Q value and target Q value of the current agent and other agents are fed into the ``Mixer`` to get the :math:`Q_{tot}`.


.. figure:: ../images/qmix.png
    :width: 600
    :align: center

    Monotonic Value Function Factorisation (QMIX)


Characteristic
^^^^^^^^^^^^^^^

action space

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``discrete``


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

   * - ``off-policy``
     - ``stochastic``
     - ``value decomposition``


Algorithm Insights
^^^^^^^^^^^^^^^^^^^^^^^

Preliminary

- :ref:`IQL`
- :ref:`VDN`

VDN optimizes multiple agents' joint policy by a straightforward operation: sum all the rewards. However, this operation reduces the
representation of the strategy because the full factorization is not necessary for extracted decentralized
policies to be entirely consistent with the centralized counterpart.

Simply speaking, VDN force each agent to find the best action to satisfy the following equation:

.. math::

    \underset{\mathbf{u}}{\operatorname{argmax}}\:Q_{tot}(\boldsymbol{\tau}, \mathbf{u}) =
    \begin{pmatrix}
    \underset{u^1}{\operatorname{argmax}}\:Q_1(\tau^1, u^1)   \\
    \vdots \\
    \underset{u^n}{\operatorname{argmax}}\:Q_n(\tau^n, u^n) \\
    \end{pmatrix}

QMIX claims that a larger family of monotonic functions is sufficient for factorization (value decomposition) but not necessary to satisfy the above equation
The monotonic constraint can be written as:

.. math::
    \frac{\partial Q_{tot}}{\partial Q_a}  \geq 0,~ \forall a \in A

With monotonic constraints, we need to introduce a feed-forward neural network that
takes the agent network outputs as input and mixes them monotonically.
To satisfy the monotonic constraint, the weights (but not the biases) of the mixing network are restricted
to be non-negative.

This neural network is named **Mixer**.

The similarity of QMIX and VDN:

- Each agent is still a standard Q function, use self-observation as input and output the action logits(Q value).
- Using standard DQN to optimize the Q function using :math:`Q_{tot}` with the team reward :math:`r`.

Difference:

- Additional model **Mixer** is added into QMIX.
- The Q values of all agents are fed to the **Mixer** for getting :math:`Q_{tot}`.
- The gradient each Q function received is backpropagated from the **Mixer**.

Similar to VDN, QMIX is only applicable to the cooperative multi-agent task.
Sharing the parameter is the primary option, which brings higher data efficiency and smaller model size.

.. admonition:: You Should Know:

    Variants of QMIX are proposed, like WQMIX and Q-attention. However, in practice, a finetuned QMIX (RIIT) is all you need.


Math Formulation
^^^^^^^^^^^^^^^^^^

QMIX needs information sharing across agents. Here we bold the symbol (e.g., :math:`s` to :math:`\mathbf{s}`) to indicate that more than one agent information is contained.

Q mixing: a learnable mixer computing the global Q value by mixing all the Q values.

.. math::

    Q_{tot}(\mathbf{a}, s;\boldsymbol{\phi},\psi) = g_{\psi}\bigl(`\mathbf{s}, Q_{\phi_1},Q_{\phi_2},..,Q_{\phi_n} \bigr)

Q learning: every iteration get a better total global Q value estimation, passing gradient to both mixer and each Q function to update them.

.. math::

    L(\phi, {\mathcal D}) = \underset{\tau \sim {\mathcal D}}{{\mathrm E}}\Bigg(Q_{\phi}^{tot} - \left(r + \gamma (1 - d) Q_{\phi_{targ}}^{tot^{'}} \right) \Bigg)^2


Here :math:`{\mathcal D}` is the replay buffer, which can be shared across agents.
:math:`r` is the reward.
:math:`d` is set to 1(True) when an episode ends else 0(False).
:math:`{\gamma}` is discount value.
:math:`Q_{\phi}` is Q function, which can be shared across agents.
:math:`Q_{\phi_{\text{targ}}}` is target Q function, which can be shared across agents.
:math:`g_{\psi}` is mixing network.
:math:`g_{\psi_{\text{targ}}}` is target mixing network.


Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We use vanilla QMIX implementation of RLlib, but with further improvement to ensure the performance is aligned with the official implementation.
The differences between ours and vanilla QMIX can be found in

- ``episode_execution_plan``
- ``EpisodeBasedReplayBuffer``
- ``JointQLoss``
- ``JointQPolicy``

Key hyperparameters location:

- ``marl/algos/hyperparams/common/qmix``
- ``marl/algos/hyperparams/finetuned/env/qmix``

Read List
-------------

- `Human-level control through deep reinforcement learning <https://daiwk.github.io/assets/dqn.pdf>`_
- `Deep Recurrent Q-learning for Partially Observable MDPs <https://www.aaai.org/ocs/index.php/FSS/FSS15/paper/download/11673/11503>`_
- `Value-Decomposition Networks For Cooperative Multi-Agent Learning <https://arxiv.org/abs/1706.05296>`_
- `QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning <https://arxiv.org/abs/1803.11485>`_
