.. _algorithms:


*******************************
MARLlib Framework
*******************************

Etiam turis ante, luctus sed velit tristique, finibus volutpat dui. Nam sagittis vel ante nec malesuada.
Praesent dignissim mi nec ornare elementum. Nunc eu augue vel sem dignissim cursus sed et nulla.
Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas.
Pellentesque dictum dui sem, non placerat tortor rhoncus in. Sed placerat nulla at rhoncus iaculis.

.. contents:: :depth: 3


Training Pipeline
----------------------

.. figure:: ../images/rllib_data_flow_left.png
    :align: center
    :width: 600

    Pre-learning Stage

Etiam turis ante, luctus sed velit tristique, finibus volutpat dui. Nam sagittis vel ante nec malesuada.
Praesent dignissim mi nec ornare elementum. Nunc eu augue vel sem dignissim cursus sed et nulla.
Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas.
Pellentesque dictum dui sem, non placerat tortor rhoncus in. Sed placerat nulla at rhoncus iaculis.

.. figure:: ../images/rllib_data_flow_right.png
    :align: center

    Learning Stage

Etiam turis ante, luctus sed velit tristique, finibus volutpat dui. Nam sagittis vel ante nec malesuada.
Praesent dignissim mi nec ornare elementum. Nunc eu augue vel sem dignissim cursus sed et nulla.
Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas.
Pellentesque dictum dui sem, non placerat tortor rhoncus in. Sed placerat nulla at rhoncus iaculis.


Environment Interface
----------------------

.. figure:: ../images/marl_env_right.png
    :align: center
    :width: 600

    Agent-Environment Interface in MARLlib

- Ten diverse environments in one interface
- Multi-task as one task.
- Group by group interact available

Etiam turis ante, luctus sed velit tristique, finibus volutpat dui. Nam sagittis vel ante nec malesuada.
Praesent dignissim mi nec ornare elementum. Nunc eu augue vel sem dignissim cursus sed et nulla.
Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas.
Pellentesque dictum dui sem, non placerat tortor rhoncus in. Sed placerat nulla at rhoncus iaculis.


Algorithms Pipeline
----------------------

Independent Learning
^^^^^^^^^^^^^^^^^^^^

.. figure:: ../images/IL.png
    :align: center
    :width: 400

    Learning pipeline of independent learning

Etiam turis ante, luctus sed velit tristique, finibus volutpat dui. Nam sagittis vel ante nec malesuada.
Praesent dignissim mi nec ornare elementum. Nunc eu augue vel sem dignissim cursus sed et nulla.
Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas.
Pellentesque dictum dui sem, non placerat tortor rhoncus in. Sed placerat nulla at rhoncus iaculis.


Centralized Critic
^^^^^^^^^^^^^^^^^^^^


.. figure:: ../images/CC.png
    :align: center
    :width: 400

    Learning pipeline of centralized critic under CTDE framework

Etiam turis ante, luctus sed velit tristique, finibus volutpat dui. Nam sagittis vel ante nec malesuada.
Praesent dignissim mi nec ornare elementum. Nunc eu augue vel sem dignissim cursus sed et nulla.
Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas.
Pellentesque dictum dui sem, non placerat tortor rhoncus in. Sed placerat nulla at rhoncus iaculis.

Value Decomposition
^^^^^^^^^^^^^^^^^^^^

.. figure:: ../images/VD.png
    :align: center
    :width: 400

    Learning pipeline of value decomposition under CTDE framework

Etiam turis ante, luctus sed velit tristique, finibus volutpat dui. Nam sagittis vel ante nec malesuada.
Praesent dignissim mi nec ornare elementum. Nunc eu augue vel sem dignissim cursus sed et nulla.
Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas.
Pellentesque dictum dui sem, non placerat tortor rhoncus in. Sed placerat nulla at rhoncus iaculis.


Available Algorithms Checklist
-------------------------------

- Independent Learning
    - :ref:`IQL`
    - :ref:`IPG`
    - :ref:`IA2C`
    - :ref:`IDDPG`
    - :ref:`ITRPO`
    - :ref:`IPPO`
- Centralized Critic
    - :ref:`COMA`
    - :ref:`MAA2C`
    - :ref:`MADDPG`
    - :ref:`MATRPO`
    - :ref:`MAPPO`
    - :ref:`HATRPO`
    - :ref:`HAPPO`
- Value Decomposition
    - :ref:`VDN`
    - :ref:`QMIX`
    - :ref:`FACMAC`
    - :ref:`VDA2C`
    - :ref:`VDPPO`

Available Environment Checklist
-------------------------------

Please refer to :ref:`env`



