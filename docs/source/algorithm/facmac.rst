.. _FACMAC:

Factored Multi-Agent Centralised Policy Gradients (FACMAC)
-------------------------------------------------------------

.. admonition:: Quick Facts

    - Factored Multi-Agent Centralised Policy Gradients (FACMAC) is one of the centralized extensions of :ref:`IDDPG`.
    - Agent architecture of FACMAC consists of three modules: ``policy``, ``Q``, and ``mixer``.
    - Policies only use local information at execution time.
    - FACMAC applies to cooperative task mode only.


Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :ref:`IDDPG`
- :ref:`QMIX`
- :ref:`VDN`

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

related algorithms

.. list-table::
   :widths: 25 25
   :header-rows: 0

   * - :ref:`IDDPG`
   * - :ref:`MADDPG`



Algorithm Insights
^^^^^^^^^^^^^^^^^^^^^^^

FACMAC is a variant of :ref:`IDDPG` in value decomposition method, and a counterpart of :ref:`MADDPG`.
The main contribution of FACMAC is:

#. First value decomposition method in MARL that can deal with continues control problem.
#. Proposed with a multi-agent benchmark :ref:`MaMujoco` that focus on continues control with heterogeneous agents.
#. Can also be applied to discrete action space with tricks like `gumble softmax` and keep robust performance

Compared to existing methods, FACMAC:

- outperforms MADDPG and other baselines in both discrete and continuous action tasks.
- scales better as the number of agents (and/or actions) and the complexity of the task increases.
- proves that factoring the critic can better take advantage of our centralised gradient estimator to optimise the agent policies when the number of agents and/or actions is large.

.. admonition:: Some Interesting Facts

    - Recent works prove that stochastic policy gradient methods are more stable and good-performance in tackling MARL. E.g., :ref:`MAA2C`. If you need better performance, try stochastic policy gradient methods.
    - Applicable scenarios of FACMAC are quite restrained. E.g., cooperative task only, continues task only(with out adding tricks).
    - Benchmarks on continues control is relatively rare in MARL. If you try to prove that your methods is good on multi-agent continues control problem, consider these benchmarks:
        - :ref:`MPE` (discrete+continues)
        - :ref:`MaMujoco` (continues only)
        - :ref:`MetaDrive` (continues only)

Math Formulation
^^^^^^^^^^^^^^^^^^

Q mixing:

.. math::

    Q_{tot}(\mathbf{a}, s;\boldsymbol{\phi},\psi) = g_{\psi}\bigl(s, Q_{\phi_1},Q_{\phi_2},..,Q_{\phi_n} \bigr)

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
:math:`d` is set to `1`(True) when an episode ends else `0`(False).
:math:`{\gamma}` is discount value.
:math:`\mu_{\theta}` is policy net, which can be shared across agents.
:math:`Q_{\phi}` is Q net, which can be shared across agents.
:math:`g_{\psi}` is mixing network.
:math:`\mu_{\theta_{\text{targ}}}` is target policy net, which can be shared across agents.
:math:`Q_{\phi_{\text{targ}}}` is target Q net, which can be shared across agents.
:math:`g_{\psi_{\text{targ}}}` is target mixing network.

.. admonition:: You Should Know

    The policy inference procedure of FACMAC is kept the same as IDDPG.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the sampling stage, each agent follows the standard DDPG learning pipeline to infer the action and send the action to Q function to get the Q value. Data like observation/state is shared among agents
before sending the sampled data to the buffer.
In the learning stage, each agent predicts its Q value using the Q function, next action using the target policy,  and next Q value using the target Q function.
Then each agent shares the predicted data with other agents before entering the training loop.

.. figure:: ../images/facmac.png
    :width: 600
    :align: center

    Factored Multi-Agent Centralised Policy Gradients (FACMAC)

.. admonition:: You Should Know

    Some tricks like `gumble_softmax` enables FACMAC net to output categorical-like action distribution.

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

Usage & Limitation
^^^^^^^^^^^^^^^^^^^^^^

FACMAC in *MARLlib* is suitable for

- continues control tasks
- cooperative tasks

.. code-block:: shell

    python marl/main.py --algo_config=facmac --finetuned --env-config=mamujoco with env_args.map_name=2AgentAnt


Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `FACMAC: Factored Multi-Agent Centralised Policy Gradients <https://arxiv.org/pdf/2003.06709.pdf>`_
