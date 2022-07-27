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

inherited algorithms

.. list-table::
   :widths: 25
   :header-rows: 0

   * - IDDPG

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Deep Deterministic Policy Gradient(DDPG).

Algorithm Description
^^^^^^^^^^^^^^^^^^^^^^^

Multi-agent Deep Deterministic Policy Gradient (MADDPG) is an algorithm extends DDPG with a centralied Q function that takes not only observation and action from current agent,
but also other agents. Similiar to DDPG, MADDPG also has a policy network :math:`\mu(s)` parameterized by :math:`\theta` to produce action value.
While the centralized Q value is calculated as :math:`Q(\mathbf{s},\mu(\mathbf{s}))` and the Q network is parameterized by :math:`\phi`.
Note :math:`s` in policy network is the self observation/state while :math:`\mathbf{s}` in centralized Q is the joint observation/state which also includes the opponents.

Math Formulation
^^^^^^^^^^^^^^^^^^

Q learning:

.. math::

    L(\phi, {\mathcal D}) = \underset{(\mathbf{s},\mathbf{a},r,\mathbf{s'},d) \sim {\mathcal D}}{{\mathrm E}}\left[
        \Bigg( Q_{\phi}(\mathbf{s},\mathbf{a}) - \left(r + \gamma (1 - d) Q_{\phi_{\text{targ}}}(\mathbf{s'}, \mu_{\theta_{\text{targ}}}(\mathbf{s'})) \right) \Bigg)^2
        \right]


Policy learning:

.. math::

    \max_{\theta} \underset{s \sim {\mathcal D}}{{\mathrm E}}\left[ Q_{\phi}(s,\mathbf{a}, \mu_{\theta}(s)) \right]

Here :math:`{\mathcal D}` is the replay buffer, which can be shared across agents.
:math:`\mathbf{a}` is an action set, including opponents.
:math:`r` is the reward.
:math:`\mathbf{s}` is the observation/state set, including opponents.
:math:`\mathbf{s'}` is the next observation/state set, including opponents.
:math:`d` is set to ``1``(True) when episode ends else ``0``(False).
:math:`{\gamma}` is discount value.
:math:`\mu_{\theta_{\text{targ}}}` is policy target net, which can be shared across agents.
:math:`\phi_{\text{targ}}` is Q target net, which can be shared across agents.

.. admonition:: You Should Know

    Policy inference procedure of MADDPG is kept same with IDDPG. While the learning target of policy net is different.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In sampling stage, each agent follows the standard DDPG learning pipeline to inference the action but use a centralized Q function to compute Q value, which needs data sharing
before send all the collected data to the buffer.
In learning stage, each agent predict its next action use target policy and share with other agents before entering the training loop.

.. figure:: ../images/MADDPG.png
    :width: 600
    :align: center

    Multi-agent Deep Deterministic Policy Gradient (MADDPG)

Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We extend vanilla DDPG of RLlib to be recurrent neural network(RNN) compatiable.
Based on RNN compatiable DDPG, we add the centralized sampling and training module to the original pipeline.
The main differences between IDDPG and MADDPG are:

- model side: the agent model related modules and functions are built in centralized style:
    - ``build_maddpg_models_and_action_dist``
    - ``MADDPG_RNN_TorchModel``
- algorithm side: the sampling and training pipelines are built in centralized style:
    - ``centralized_critic_q``
    - ``central_critic_ddpg_loss``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/maddpg``
- ``marl/algos/hyperparams/fintuned/env/maddpg``

Usage & Limitation
^^^^^^^^^^^^^^^^^^^^^^

MADDPG is only suitable for

- continues control tasks.

.. code-block:: shell

    python marl/main.py --algo_config=maddpg --finetuned --env-config=mamujoco with env_args.map_name=2AgentAnt



Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments <https://arxiv.org/abs/1706.02275>`_
