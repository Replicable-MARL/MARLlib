.. _concept:

***************************************
Benchmarks
***************************************

We collect most of the existing MARL benchmarks here with brief introduction.

.. contents::
    :local:
    :depth: 3

PyMARL
========================

Github: https://github.com/oxwhirl/pymarl

PyMARL is the first and most well-known MARL library. All algorithms in PyMARL is built for SMAC, where agents learn to cooperate for a higher team reward. However, PyMARL has not been updated for a long time,
and can not catch up with the recent progress. To address this, the extension versions of PyMARL are presented including PyMARL2 and EPyMARL.


PyMARL2
========================

Github: https://github.com/hijkzzz/pymarl2

PyMARL2 focuses on credit assignment mechanism and provide a finetuned QMIX with state-of-art-performance on SMAC.
The number of available algorithms increases to ten, with more code-level tricks incorporated.


EPyMARL
========================

Github: https://github.com/uoe-agents/epymarl

EPyMARL is another extension for PyMARL that aims to present a comprehensive view on how to unify cooperative MARL algorithms.
It first proposed the independent learning, value decomposition, and centralized critic categorization, but is restricted to cooperative algorithms. Nine algorithms are implemented in EPyMARL.
Three more cooperative environments LBF, RWARE, and MPE are incorporated to evaluate the generalization of the algorithms.

MARL-Algorithms
========================

Github: https://github.com/starry-sky6688/MARL-Algorithms

MARL-Algorithm is a library that covers broader topics compared to PyMARL including learning better credit assignment, communication-based learning,
graph-based learning, and multi-task curriculum learning. Each topic has at least one algorithm, with nine implemented algorithms in total. The testing bed is limited to SMAC.

MAPPO benchmark
========================

Github: https://github.com/marlbenchmark/on-policy

MAPPO benchmark is the official code base of MAPPO. It focuses on cooperative MARL and covers four environments. It aims at building a strong baseline and only contains MAPPO.

MAlib
========================

Github: https://github.com/sjtu-marl/malib

MAlib is a recent library for population-based MARL which combines game-theory and MARL algorithm to solve multi-agent tasks in the scope of meta-game.

MARLlib
========================

Please refer to :ref:`intro`.



