.. _part3:

********************************************************************
Part 3. A Collective Survey of MARL
********************************************************************

.. contents::
    :local:
    :depth: 3


Tasks: Arenas of MARL
=====================

Finding a dataset to evaluate the new idea has become a consensus in machine learning.
The best way is to get a representative or commonly acknowledged dataset,
follow its standard evaluation pipeline, and compare the performance with other existing algorithms.

In MARL, the dataset is a set of scenarios contained in one multi-agent task.
Most multi-agent tasks are customizable on many aspects like agent number, map size, reward function, unit status, etc.
In this section, we briefly introduce the category of multi-agent tasks, from the most straightforward matrix game
to the real-world application.

Matrix Problem and Grid World
--------------------------------------------------------------

.. figure:: ../images/twostep.jpg
    :align: center

    Two-step game

The first option is using a matrix and grid world task to verify a new idea in MARL quickly.
A well-known example is the **Two-step Game**.
In this task, two agents act in turn to gain the highest team reward.
The task is very straightforward:

#. two agents in the task
#. the observation is a short vector with a length four
#. two actions (A&B) to choose from

Despite the simple task setting, however, the game is still very challenging as one agent needs to coordinate with another agent
to achieve the highest reward: the joint action with the highest reward is not a good option from the view of the first agent if it is not willing to cooperate with another agent.
**Two-step Game** evaluates whether an agent has learned to cooperate by sacrificing its reward for a higher team reward.

As the value(reward) of the matrix can be customized, the number of matrix combinations (scenarios) that can be solved is a good measurement of the robustness of an algorithm in solving **cooperative-like** multi-agent tasks.

The grid world-based tasks are relatively more complicate than the matrix problem.
A well-known grid world example in RL is `frozen lake <https://towardsdatascience.com/q-learning-for-beginners-2837b777741>`_.
For MARL, there are many grid-world-based tasks, including:

- :ref:`LBF`
- :ref:`RWARE`
- :ref:`MAgent`

Different tasks target different topics like mixed cooperative-competitive task mode, sparse reward in MARL, and many agents in one system.

Gaming and Physical Simulation
--------------------------------------------------------------

.. figure:: ../images/gaming.jpg
    :align: center

    Gaming & Simulation: MAMuJoCo, Pommerman, Hanabi, Starcraft, etc.

To find a offset between naive matrix games and the expensive cost of sampling and training on real-world scenarios,  recent MARL research focuses more on video gaming and physical simulation,
as most algorithms try to prove their advance on more complicated tasks with a modest cost.
One of the most popular multi-agent tasks in MARL is StarCraft Multi-Agent Challenge(:ref:`SMAC`), which is for discrete control and cooperative task mode.
For continuous control, the most used task is the multi-agent version of MuJoCo: (:ref:`MAMuJoCo`).
To analyze the agent behavior of adversary agents, a typical task is :ref:`Pommerman`.
Scenarios within one task can contain different task modes, like :ref:`MPE`, which simplifies the evaluation procedure of the algorithm's generalization ability within one task domain.


Towards Real-world Application
--------------------------------------------------------------

.. figure:: ../images/realworld.jpg
    :align: center

    Real World Problem: MetaDrive, Flatland, Google Research Football, etc.

Tasks that are real-world-problem oriented, including traffic system design(:ref:`MetaDrive`), football(:ref:`Football`), and auto driving, also benchmark
recent years' MARL algorithms. These tasks can
inspire the next generation of AI solutions.
Although the tasks belonging to this categorization are of great significance to the real application, unluckily, fewer algorithms choose to be built on
these tasks due to high complexity and standard evaluation procedure.


Methodology of MARL: Task First or Algorithm First
====================================================================

Current research on MARL is struggling with the diversity of multi-agent tasks and the categorization of MARL algorithms.
These characteristics make the fair comparison of different algorithms hard to conduct and throw a question to researchers: should algorithms be developed for one task (task first)
or for general tasks (algorithm first)
This is partly due to multi-agent tasks,
as well as the various learning styles and knowledge-sharing strategies.

As the algorithm development is bound tightly with the task features, we can see an offset between the algorithm's generalization on
a broad topic and its expertise in one particular multi-agent task.

In the following part, we first briefly introduce how the environment is categorized according to the agents' relationship.
Then we categorize the algorithms depending on their learning style and how the learning style is connected to the agents' relationship.

Finally, we will talk about extending MARL algorithms to be more general and applicable to real-world scenarios via knowledge-sharing techniques.


Agent Relationship
--------------------------------------------------------------

.. figure:: ../images/relation.png
    :align: center


The relationship among agents regulates agent learning.
Two aspects of this relationship affect the MARL algorithm development the most.

First is the **working mode** of agents. For example, a task can be Cooperative-like, where agents share the same target.
A task can be Competitive-like, where agents have different or adversary targets.
We also refer **working mode** as **task mode**, as an overall description of how agents work and learn in a multi-agent task.

The second is agent similarity. A task can contain homogeneous agents which prefer knowledge sharing with others and learning as a team.
A task can also contain heterogeneous agents, which prefer learning their policies separately.

Task Mode: Cooperative-like or Competitive-like
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The task modes can be roughly divided into two types: **cooperative-like**, where agents tend to work as a team, and **competitive-like**, where agents have adversarial targets and are aggressive to other agents.


Mode 1: Cooperative-like
"""""""""""""""""""""""""""""

Cooperative-like task mode is commonly seen in many scenarios where agents are only awarded when the team target is met.
This mode is a strict **cooperative**, where each agent cannot access its own reward.
**cooperative** tasks require agents to have a robust credit assignment mechanism to decompose the global reward to update their policies.

Environments contain **cooperative** scenarios:

- :ref:`SMAC`
- :ref:`MAMuJoCo`
- :ref:`Football`
- :ref:`MPE`
- :ref:`LBF`
- :ref:`RWARE`
- :ref:`Pommerman`

Another mode is **collaborative**, where agents can access individual rewards. Under this mode, the agents tend to work together, but the target varies between different agents.
Sometimes individual rewards may cause some potential interest conflict.
Collaborative task mode has less restriction and richer reward information for wilder algorithms development:
:ref:`il` is a good solution for collaborative tasks, as each agent has been allocated an individual reward for doing a standard RL.
:ref:`cc` is a more robust algorithm family for collaborative tasks as the improved critic help agent coordinate using global information.
:ref:`vd`-based methods are still applicable for collaborative tasks as we can integrate all the individual rewards received into one (only the agents act simultaneously).
**Cooperative** mode can also be transformed to **collaborative** as we can copy the global reward to each agent and treat them as an individual reward.

Environments contain **collaborative** scenarios:

- :ref:`SMAC`
- :ref:`MAMuJoCo`
- :ref:`Football`
- :ref:`MPE`
- :ref:`LBF`
- :ref:`RWARE`
- :ref:`Pommerman`
- :ref:`MAgent`
- :ref:`MetaDrive`
- :ref:`Hanabi`

Mode 2: Competitive-like
""""""""""""""""""""""""""""""

Once agents have different targets in one task, especially when the targets are adversaries,
the task can become much more complicated. A famous example is **zero-sum** game, where the total reward is fixed.
One agent being rewarded will result in another agent being punished.
A specific example can be found in :ref:`MPE` that in scenarios like **simple_push**, agent ONE is trying to gain more reward by
getting closer to its target location while agent TWO gains reward by pushing agent ONE away from the target location.

Moreover, the competitive-like mode can also be not so **pure competitive**. It can incorporate some cooperative agents' relationships.
This type of work mode is referred to as **mixed** mode. A representative task of mixed mode is :ref:`MAgent`, where agents are divided into several groups. Agents in the same group need to attack the enemy group cooperatively.

Environments contain **competitive** or **mixed** scenarios:

- :ref:`MPE`
- :ref:`Pommerman`
- :ref:`MAgent`

Agents Type: Heterogeneous or Homogeneous
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two methods exist to solve the multi-agent problem, **heterogeneous** and **homogeneous**. Homogeneous agent affiliated with the environment holds the same policy. The policy gives out different actions based on the agent's observation.
Heterogeneous methods require each agent to maintain its individual policy, which can accept different environment observation dimensions or output actions with diverse semantic meanings.

Learning Style
--------------------------------------------------------------

Categorizing the MARL algorithm by its learning style provides an overview of which topic researchers are most interested in.
The first class is **Independent Learning**, which directly applies single-agent RL to multi-agent settings.
The second class is **Centralized Training Decentralized Execution**, where extra modules are added to the training pipeline
to help agents learn a coordinated behavior while keeping an independently executed policy.
The third class is **Fully Centralized**, where agents are treated as a single agent with multiple actions to execute simultaneously.

Independent Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We then get an independent policy to separate one agent from the multi-agent system and train this agent using RL ignoring other agents and system states. This is the core idea of independent learning. Based on this, if every agent learns its policy independently,
we can have a group of policies that jointly solve the task.

Every RL algorithm can be extended to be MARL compatible, including:

- :ref:`IQL`
- :ref:`IA2C`
- :ref:`IDDPG`
- :ref:`IPPO`
- :ref:`ITRPO`

However, independent learning always falls into the local-optimal, and performance degrades rapidly when the multi-agent tasks require
a coordinated behavior among agents. This is primarily due to the low utilization of other agents' information and the system's global state.


Centralized Training Decentralized Execution (CTDE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To help agents learn a coordinated behavior while keeping a low computation budget and optimization complexity, many different learning settings have been proposed,
among which the Centralized Training Decentralized Execution (CDTE) framework has attracted the most attention in recent years MARL research.
We have introduced the CTDE framework here: :ref:`CTDE`.

Under the CTDE framework, there are two main branches of algorithms: **Centralized Critic (CC)** and **Value Decomposition (VD)**.
The CC-based algorithms can cover general multi-agent tasks while having some restrictions on their architecture.
The VD-based algorithms are good at solving cooperative-like tasks with the credit assignment mechanism, while the tasks they can cover are limited.

Type 1. Centralized Critic
"""""""""""""""""""""""""""

CC is first used in MARL since the :ref:`MADDPG`.
As the name indicated, a critic is a must in a CC-based algorithm, which excludes most Q-learning-based algorithms as they have no
critic module. Only actor-critic algorithms like :ref:`MAA2C` or actor-Q architecture like :ref:`MADDPG` fulfill this requirement.

For the training pipeline of CC, the critic is targeting finding a good mapping between the value function and the combination of system state and self-state.
This way, the critic is updated regarding the system state and the local states.
The policy is optimized using policy gradient according to GAE produced by the critic.
The policy only takes the local states as input to conduct a decentralized execution.

The core idea of CC is to provide different information for critics and policy to update them differently.
The critic is centralized as it utilizes all the system information to accurately estimate the whole multi-agent system.
The policy is decentralized, but as the policy gradient comes from the centralized critic,
it can learn a coordinated strategy.

A list of commonly seen CC algorithms:

- :ref:`MAA2C`
- :ref:`COMA`
- :ref:`MADDPG`
- :ref:`MATRPO`
- :ref:`MAPPO`
- :ref:`HATRPO`
- :ref:`HAPPO`

Type 2. Value Decomposition
""""""""""""""""""""""""""""""

VD is introduced to MARL since the :ref:`VDN`.
The name **value decomposition** is based on the fact that the value function of each agent is updated by factorizing the global value function.
Take the most used baseline algorithms of VD :ref:`VDN` and :ref:`QMIX` for instance: VDN sums all the individual value functions to get the global function.
QMIX mixes the individual value function and sets non-negative constraints on the mixing weight.
The mixed global value function can then be optimized to follow standard RL. Finally, if learnable, backpropagated gradient updates all the individual value functions and the mixer.

Although VDN and QMIX are all off-policy algorithms, the value decomposition can be easily transferred to on-policy algorithms like :ref:`VDA2C` and :ref:`VDPPO`.
Instead of decomposing the Q value function, on-policy VD algorithms decompose the critic value function. And using the decomposed individual critic function to update the
policy function by policy gradient.

The pipeline of the VD algorithm is strictly CTDE. Global information like state and other agent status is only accessible in the mixing stage in order to maintain a decentralized policy or
individual Q function.

A list of commonly seen VD algorithms:

- :ref:`VDN`
- :ref:`QMIX`
- :ref:`FACMAC`
- :ref:`VDA2C`
- :ref:`VDPPO`


Fully Centralized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A fully centralized method is an option when the agent number and the action space are relatively small.
The approach of the fully centralized algorithm to the multi-agent tasks is straightforward: combine all the agents and their action spaces into one and follow a standard RL
pipeline to update the policy or Q value function.
For instance, a five-agent discrete control problem can be transformed into a single-agent multi-discrete control problem.
Therefore, only a cooperative-like task mode is suitable for this approach. It would be nonsense to combine agents that are adversaries to each other.

Few works focus on fully centralized MARL. However, it can still serve as a baseline for algorithms of CTDE and others.


Knowledge Sharing
--------------------------------------------------------------

Agents can share the knowledge with others to learn faster or reuse the knowledge from the old task to adapt quickly to new tasks.
We can quickly get this inspiration based on the fact that different strategies may share a similar function.
Moreover, this similarity exists across three levels in MARL: agent, scenario, and task.
Agent-level knowledge sharing is targeted to increase sample efficiency and improve learning speed.
Scenario-level sharing focuses on developing a multi-task MARL to handle multiple scenarios simultaneously but in the same task domain.
Task-level sharing is the most difficult, and it requires an algorithm to learn and conclude general knowledge from one task domain and apply them to
a new domain.

Agent Level
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Agent-level knowledge sharing mainly focuses on two parts: replay buffer and model parameters.
In most cases, these two parts are bound, meaning if two agents share the model parameters, they share the replay buffer(sampled data in the on-policy case).
There are still some exceptions, like only part of the model is shared. For instance,  in actor-critic architecture, if only the critic is shared, then the critic is
updated with the full data while the policy is updated with the data sampled.

Knowledge across other agents can significantly benefit the algorithm performance as it improves the sample efficiency and thus can be an essential trick in MARL.

However, sharing knowledge is not always good. For example, in some circumstances, we may need a diverse individual policy set, but the sharing operation vastly reduces this diversity.
An extreme instance will be adversary agents who never share knowledge to keep competitiveness.

Scenario Level
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scenario-level multi-task MARL focuses on learning a general policy that can cover multiple scenarios in one task. Therefore, scenario-level multi-task MARL is of great feasibility than task-level multi-task MARL
as the learned strategies of different scenarios are more similar than different.
For instance, although scenarios in SMAC vary on unit type, agent number, and map terrain, skills like hit and run always exist in most of the learned
strategy from them.

Recent work has proved that scenario-level knowledge sharing is doable with transformer-based architecture and a meta-learning method.
This is a potential solution for real-world applications where the working environment constantly changes and requires agents to adapt to new scenarios soon.

Task Level
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Task level multi-task MARL is the final step of learning a self-contained and constantly evolving strategy, with no restrictions on task mode and easily
adopting new tasks and reusing the knowledge from other tasks.

Task-level knowledge sharing requires agents to conclude common sense from different tasks.
For instance, when a new cooperative-like task comes, agents behave more agreeably with others and can quickly find a way to cooperate as a team.
As common sense and team-working concepts are what make human beings intelligent, achieving task-level knowledge-sharing equals training an agent
to learn and act like humans is the holy grail of artificial general intelligence.






















