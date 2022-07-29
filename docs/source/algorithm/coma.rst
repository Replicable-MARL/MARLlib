.. _COMA:

Counterfactual Multi-Agent Policy Gradients (COMA)
-----------------------------------------------------

.. admonition:: Quick Facts

    - Counterfactual Multi-Agent Policy Gradients(COMA) is one of the earliest works on centralized critic MARL.
    - Agent architecture of COMA consists of two modules: policy network and critic network.
    - COMA adopt a new credit assignment mechanism that use counterfactual baseline to marginalises out a single agent’s action.
    - COMA is surprisingly good in multi-agent tasks where different actions have significant different impact to the system.

Preliminary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`IA2C`

Characteristic
^^^^^^^^^^^^^^^

action space:

.. list-table::
   :widths: 25
   :header-rows: 0

   * - ``discrete``

task mode:

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

Efficiently learning decentralised policies is essential demand for modern AI systems. When there is only one reward, how to assign the credit to agent becomes a major challenge.
COMA provide one solution for this problem:

#. COMA uses a counterfactual baseline that marginalises out a single agent’s action, while keeping the other agents’ actions fixed.
#. COMA develops a centralized critic that allows the counterfactual baseline to be computed efficiently in a single forward pass.
#. COMA significantly improves average performance over other multi-agent actor critic methods under decentralised execution and partial observability settings.

.. admonition:: Some Interesting Facts

    - Although COMA is based on stochastic policy gradient methods, it is only used in discrete action space. Extending to continues action space require additional tricks on computing critic value (which is not a good news for stochastic methods)
    - In recent years research, COMA's is proven to relatively worse in solving tasks like :ref:`SMAC` and :ref:`MaMujoco`, even compared to basic independent actor critic methods like :ref:`IA2C`.
    - Give COMA a chance when your SOTA methods struggling in learning a good policy. *It may surprise you*.

Math Formulation
^^^^^^^^^^^^^^^^^^

Critic learning:

.. math::

    \phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( Q_{\phi} (s_t, a_t) - \hat{R}_t \right)^2

Marginalized Advantage Estimation:

.. math::

    A(s, \mathbf{s}^-, a, \mathbf{a}^-) = Q(s, \mathbf{s}^-, a, \mathbf{a}^-) - \sum_{a'} \pi(a' \vert \tau) Q(s,(\mathbf{a}^{-},a'))


Policy learning:

.. math::

    L(s,\mathbf{s}^-,a, \mathbf{a}^-, \theta)=\log\pi_\theta(a|s)A(s, \mathbf{s}^-, a, \mathbf{a}^-)

Here
:math:`{\mathcal D}` is the collected trajectories.
:math:`R` is the rewards-to-go.
:math:`\tau` is the trajectory.
:math:`A` is the advantage.
:math:`a` is the current agent action.
:math:`\mathbf{a}^-` is the action set of all agents, except the current agent.
:math:`s` is the current agent observation/state.
:math:`\mathbf{s}^-` is the observation/state set of all agents, except the current agent.
:math:`\epsilon` is a hyperparameter controlling how far away the new policy is allowed to go from the old.
:math:`Q_{\phi}` is the Q value function.
:math:`\pi_{\theta}` is the policy net.


Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the sampling stage, agents share information with others. The information includes others' observations and predicted actions. After collecting the necessary information from other agents,
all agents follow the standard PPO training pipeline, except using the centralized critic value function to calculate the GAE and conduct the PPO critic learning procedure.

.. figure:: ../images/coma.png
    :width: 600
    :align: center

    Counterfactual Multi-Agent Policy Gradients (COMA)

Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

We use vanilla A2C implementation of RLlib in COMA as the base of IA2C.
COMA is further based on IA2C. We add centralized modules and a COMA-loss function to implement COMA.
The main differences are:

    - ``centralized_critic_postprocessing``
    - ``central_critic_coma_loss``
    - ``CC_RNN``


Key hyperparameter location:

- ``marl/algos/hyperparams/common/coma``
- ``marl/algos/hyperparams/fintuned/env/coma``

Usage & Limitation
^^^^^^^^^^^^^^^^^^^^^^

COMA in *MARLlib* is suitable for

- discrete control tasks
- any task mode

.. code-block:: shell

    python marl/main.py --algo_config=coma --finetuned --env-config=smac with env_args.map_name=3m

Read list
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `Counterfactual Multi-Agent Policy Gradients <https://ojs.aaai.org/index.php/AAAI/article/download/11794/11653>`_
