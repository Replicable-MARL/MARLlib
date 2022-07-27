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

    \max_{\theta} \underset{s \sim {\mathcal D}}{{\mathrm E}}\left[ Q_{\phi}(s, \mu_{\theta}(s)) \right]

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

    Policy inference of MADDPG is exactly same as DDPG/IDDPG. While the optimization of policy net is different.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In sampling stage, each agent follows the standard DDPG learning pipeline to inference the action but use a centralized Q function to compute Q value, which needs data sharing.
In learning stage, each agent predict its next action use target policy and share with other agents before entering the training loop.

.. figure:: ../images/MADDPG.png
    :width: 600
    :align: center

    Multi-agent Deep Deterministic Policy Gradient (MADDPG)

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments <https://arxiv.org/abs/1706.02275>`_
