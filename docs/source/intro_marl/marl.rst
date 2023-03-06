.. _part2:

***************************************
Part 2. Navigate From RL To MARL
***************************************

.. contents::
    :local:
    :depth: 3

RL, especially Deep RL's remarkable success in solving decision-making problems, is vital to many related fields.
One of them is multi-agent reinforcement learning (MARL).
MARL focuses on both agent itself that targets learning a good strategy and the behavior of a group of agents:
how do agents coordinate when sharing the same group target or
evolve when facing adversary agents.

MARL: On the shoulder of RL
----------------------------------------

MARL is the natural extension of RL with more attention to the analysis of the above-mentioned group behavior,
the core component is kept unchanged; that is how the agent optimizes its strategy to gain a higher reward.
Most existing MARL algorithms are built on well-known RL algorithms like Q learning and Proximal Policy Optimization (PPO).
The RL algorithms underpin MARL as a strong tool such that directly applying the RL algorithm to multi-agent tasks can be empirically good in some cases of MARL. However, using plain RL algorithms with no external signal from other agents is apparently not optimal. To understand the necessity of navigating RL to MARL for multi-agent tasks, we have some basic concepts to cover.

.. _POMDP:

Partially Observable Markov Decision Process (POMDP)
--------------------------------------------------

**Partially Observable Markov Decision Process (POMDP)** is a process where the unobservable system states probabilistically to observations.
The agent cannot access the system's entire state, and the same action can incur different observations.
However, the observation is still conditioned on the system state, meaning the agent has to learn or hold a belief in its observation
and learn a policy that adopts all possible states.

POMDP is more general in modeling sequential decision-making problems, especially for multi-agent settings.
Consider a group of agents aiming at solving a complicated cooperative task.
The information of the whole system is way too much for one agent to solve the work at hand.
A more practical approach is only to provide one agent with a local observation instead of the global state to do the decision-making: select an action that only fits the current situation.

The only question remains on how to help the agent build its belief: the decision it makes based on local observation should align with both its teammates and the system state towards the final target.
One of the most commonly used techniques is **Centralized Training & Decentralized Execution (CTDE)**, which we will discuss next.

.. _CTDE:

Centralized Training & Decentralized Execution (CTDE)
-----------------------------------------------------

.. figure:: ../images/ctde.png
    :align: center

**Centralized Training & Decentralized Execution (CTDE)** is a setting where agent learn(train) together and execute(test) separately.
Specifically, there is no restriction in the training stage of CTDE, where the agent can access all the information the system can provide, including other agents' status, the global state, and even the reward other agents get.
Therefore, the training style is fully centralized.
While at the execution stage, agents are forced to make their own decision based on their observations. Any information sharing among agents and information that contains the system state is forbidden.
In this way, the execution is fully decentralized.

CTDE is the most popular framework in MARL as it finds an outstanding balance between coordination learning and deployment cost.
A multi-agent task contains numerous agents; learning a policy that fits the group target must be conditioned on the extra information from other sources. Therefore, centralized training is the best choice.
However, after we get the trained policy and deploy them, the extra information delivery is way too expensive. It can be the system's bottleneck: agents must wait for the **centralized** information to arrive before making the decision.
Centralized execution is also unsafe as the **centralized** information can be hacked and falsified while delivering, which causes a chain reaction in future decision-making.

Following the CTDE framework, there are two main branches of MARL algorithms:

- :ref:`CC`
- :ref:`VD`

A centralized critic-based algorithm is applicable for all multi-agent task modes, including cooperative, collaborative, competitive, and mixed.
While value decomposition-based algorithms can only be applied to cooperative and collaborative scenarios.

Diversity: Task Mode, Interacting Style, and Additional Infomation
----------------------------------------------------------------

Multi-agent tasks are well-known for their diversity. However, in single-agent tasks, this diversity is limited to discrete/continual action space, dense/sparse reward function, and observation dimension.
Multi-agent tasks contribute more to this diversity based on single-agent tasks, including:

#. **Task mode**: how agents work together.
#. **Interacting style**: in which order the agent interacts with the environment.
#. **Additional information**: depends on whether the environment provides it.

These three elements make multi-agent tasks far more complicated than singe agent tasks.
The large diversity also creates a unique challenge when comparing the MARL algorithms.
An algorithm that performs well in one task can not guarantee its advance in another.
Building a unified benchmark for MARL is then facing a big challenge.


What can MARL do?
----------------------------------------

We can see that MARL bridges RL and real-world scenarios in a specific way.
Teaching a group of agents how to coordinate, providing extra information to guide the strategy evolution,
and equipping agents with the ability to handle diverse tasks with a more general policy. These are the motivations of MARL,
also the target of artificial general intelligence.

MARL can now outperform humans in games like chess and `MOBA <https://en.wikipedia.org/wiki/Multiplayer_online_battle_arena>`_,
solve real-world tasks like vision+language-based navigation,
help to design a better traffic system, etc.

The increasing number of research papers and industrial applications is witnessing a new revolution in the MARL area.











