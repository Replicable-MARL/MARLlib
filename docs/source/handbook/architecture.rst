.. _algorithms:


*******************************
Framework
*******************************

Based on Ray and one of its toolkits RLlib, MARLlib enriches the RLlib with 18 multi-agent reinforcement learning (MARL) algorithms and incorporates ten diverse multi-agent environments as a testing bed.
All the algorithms can be smoothly run on any environment with auto adaptation like model architecture and environment interface, and flexible customization via simply modifying the configuration files.


.. contents::
    :local:
    :depth: 3


Architecture
====================

In this part, we introduce the MARLlib training pipelines from three perspectives:

- agent and environment interaction
- data sampling and training workflow
- core components that form the whole pipeline

Environment Interface
-----------------------

.. figure:: ../images/marl_env_right.png
    :align: center
    :width: 600

    Agent-Environment Interface in MARLlib

The environment interface in MARLlib enables the following abilities:

#. agent-agnostic: each agent has insulated data in the training stage
#. task-agnostic: diverse environments in one interface
#. asynchronous sampling: flexible agent-environment interaction mode

First, MARLlib treats MARL as the combination of single agent RL processes.

Second, MARLlib unifies all the ten environments into one abstract interface that helps the burden for algorithm design work. And the environment under this interface
can be any instance, enabling multi-tasks / task agnostic learning.

Third, unlike most of the existing MARL framework that only supports synchronous interaction between agents and environments, MARLlib supports an asynchronous interacting style.
This should be credited to RLlib's flexible data collecting mechanism as data of different agents can be collected and stored in both synchronous and asynchronous ways.


Workflow
-----------------------

Same as RLlib, MARLlib has two phases after launching the process.

Phase 1:   Pre-learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MARLlib initializes the environment and the agent model, producing a fake batch according to environment attributes and passing it to the sampling/training pipeline of the chosen algorithm.
If the fake batch goes through the whole learning workflow with no error reported, MARLlib steps into the next stage.

.. figure:: ../images/rllib_data_flow_left.png
    :align: center
    :width: 600

    Pre-learning Stage


Phase 2: Sampling & Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After checking the whole pipeline in the pre-learning stage, real jobs are assigned to the workers and the learner. Then finally, these processes are scheduled under the execution plan, where MARL officially starts.

In a standard learning iteration, each worker first samples the data by interacting with its environment instance(s) using agent model(s). Then, the workers pass The sampled data to the replay buffer.
Reply buffer is initialized according to the algorithm, which will decide how the data are stored. For example, the buffer is a concatenation operation for the on-policy algorithm.
For the off-policy algorithm, the buffer is a FIFO queue.

Next, a pre-defined policy mapping function will distribute these data to different agents.
Once the data for one training iteration is fully collected, the learner starts to optimize the policy/policies using these data
and broadcasts the new model to each worker for the next sampling round.

.. figure:: ../images/rllib_data_flow_right.png
    :align: center

    Sampling & Training Stage


Algorithm Pipeline
----------------------------------------

.. image:: ../images/pipeline.png
    :align: center

Independent Learning
^^^^^^^^^^^^^^^^^^^^

Independent learning (left) is easy to implement in MARLlib as RLlib provides many algorithms.
Choosing one from them and applied to the multi-agent environment to start training is easy and require no extra work compared to RLlib.
While no data exchange is needed in independent learning of MARL, the performance is worse than the centralized training strategy in most tasks.

Centralized Critic
^^^^^^^^^^^^^^^^^^^^

Centralized critic learning (middle) is one of the two centralized training strategies under the CTDE framework.
Agents must share their information after getting the policy output and before the critic value computing.
They must share specific information with other agents, including individual observation, actions, and global state (if available).

The exchanged data is collected and stored as transition data during the sampling stage. Each transition data contains both self-collected data and exchanged data.
All the data is then used to optimize a centralized critic function with a decentralized policy function.
How information is shared is mainly implemented in the postprocessing function for on-policy algorithms. For off-policy algorithms like MADDPG,
additional data like action value provided by other agents is collected before the data enters the training iteration batch.

Value Decomposition
^^^^^^^^^^^^^^^^^^^^

Value Decomposition (right) is another branch of centralized training strategies. Different from a centralized critic, the only information for the agent
to share is the predicted Q value or critic value. Additional data is required according to the algorithm. For instance, QMIX needs a global state to
compute the mixing Q value.

The data collecting and storage logic is the same as a centralized critic. The joint Q learning methods (VDN, QMIX) are heavily copied from the original PyMARL. Only the FACMAC, VDA2C, and VDPPO follow the standard RLlib training pipeline among all five value decomposition algorithms.


Key Component
-------------------------

Postprocessing Before Data Collection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MARL algorithms with centralized training with decentralized execution (CTDE) require agents to share their information with others in the learning stage.
Algorithms in value decomposition like QMIX, FACMAC, and VDA2C require other agents to provide their Q value or V value estimation to compute Q total or V total. Likewise, algorithms in centralized criticism like MADDPG, MAPPO, and HAPPO require other agents to provide their observation and actions to help determine a centralized critic value.
A postprocessing module is then a perfect place for agents to share the data with other agents.
For algorithms belonging to centralized critics, the agent can get extra information from other agents to compute a centralized critic value.
For algorithms belonging to value decomposition, the agent needs to provide other agents with their Q or V value predicted.
Besides, the postprocessing module is also the place for computing different learning targets using GAE or N-step reward adjustment.

.. figure:: ../images/pp.png
    :align: center

    Postprocessing Before Data Collection

Postprocessing Before Batch Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Postprocessing is unsuitable for every algorithm; exceptions are off-policy algorithms, including MADDPG and FACMAC.
The problem is that the data stored in the replay buffer are from the old model, e.g., Q value, which can not be used for the current training interaction.
To deal with this, the additional before batch learning function is adopted to calculate the accurate Q or V value
using the current model just before the sampled batch enters the training loop.

.. figure:: ../images/pp_batch.png
    :align: center

    Postprocessing Before Batch Learning


Centralized Value function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The centralized critic agent model abandons the original value function conditioned only on self-observation. Instead, a centralized critic who dynamically fits the
algorithm needs are provided to deal with data supplied from other agents and output a centralized value.

Mixing Value function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The value decomposition agent model preserves the original value function but adds a new mixing value function to get the mixing value function.
The mixing function is customizable. Currently, VDN and QMIX mixing function is provided. To change the mixing value, modify
the model configuration file in **marl/model/configs/mixer**.

Heterogeneous Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In heterogeneous optimization, the parameters of each agent are updated separately.
Therefore, policy function is not shared across different agents.
According to the proof of the algorithm, if agents were to set the values of the loss-related summons by sequentially updating their policies,
any positive update would lead to an increment in summation.

To ensure the monotonic increment, we use the trust region to get the suitable parameters update (HATRPO).
Considering the computing consumption, we use the proximal policy optimization to speed up the policy and critic update (HAPPO).

.. figure:: ../images/hetero.png
    :align: center

    Heterogeneous Agent Critic Optimization

Policy Mapping
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Policy mapping plays an important role in unifying the MARL environment interface. In MARLlib, the policy mapping is designed to be a dictionary,
with a top-level key as the scenario name, a second-level key as the group information, with four extra keys including **description**, **team_prefix**,
**all_agents_one_policy**, and **one_agent_one_policy**. **team_prefix** is used to group the agents according to their names.
The last two keys indicate whether a fully shared or no-sharing policy strategy is a valid option for this scenario.
We use policy mapping to initialize the policies and allocate them to different agents.
Each policy is optimized only using the data sampled by the agent that belongs to this policy group.

Here is an example of policy mapping, which is a mixed mode scenario from MAgent:


.. code-block:: ini

    "adversarial_pursuit": {
        "description": "one team attack, one team survive",
        "team_prefix": ("predator_", "prey_"),
        "all_agents_one_policy": False,
        "one_agent_one_policy": False,
    },


Algorithms Checklist
================================

Independent Learning
---------------------

- :ref:`IQL`
- :ref:`IPG`
- :ref:`IA2C`
- :ref:`IDDPG`
- :ref:`ITRPO`
- :ref:`IPPO`

.. _cc:

Centralized Critic
---------------------

- :ref:`MAA2C`
- :ref:`COMA`
- :ref:`MADDPG`
- :ref:`MATRPO`
- :ref:`MAPPO`
- :ref:`HATRPO`
- :ref:`HAPPO`

.. _vd:

Value Decomposition
---------------------

- :ref:`VDN`
- :ref:`QMIX`
- :ref:`FACMAC`
- :ref:`VDA2C`
- :ref:`VDPPO`

Environment Checklist
================================

Please refer to :ref:`env`



