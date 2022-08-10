.. _quick-start:

Quick Start
===========

.. contents::
    :local:
    :depth: 2

If you have not installed MARLlib yet, please refer to :ref:`basic-installation` before running.

Before Running
-----------------

MARLlib provides dozens of environments for you to choose from.
After installing :ref:`basic-installation`, you don't have to install every one of these environments.
Follow the instruction :ref:`env` to install the environment you need.


Environment Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As the :ref:`env` we incorporate in MARLlib is diverse, and each one has its unique environmental setting,
we leave a `env configure <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/envs/base_env/config>`_ directory to change the hyper-parameter passed to the environment initialization.

For instance, the Multiple Particle Environments (MPE) are set to accept only discrete action.
To allow continues action, simply change **continuous_actions** in `mpe.yaml <https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/envs/base_env/config/mpe.yaml>`_ to **True**


Algorithm Hyper-parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the environment is all set, you need to visit the MARL algorithms' hyper-parameter directory.
Each algorithm has different hyper-parameters to finetune with.

Most of the algorithms are sensitive to the environment settings.
This means you need to give a set of hyper-parameters that fit for current MARL task.

We provide a `commonly used hyper-parameters directory <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/algos/hyperparams/common>`_.
And a finetuned hyper-parameters sets for the four most used MARL environments/benchmarks, including

- `GRF <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/algos/hyperparams/finetuned/football>`_
- `SMAC <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/algos/hyperparams/finetuned/smac>`_
- `MPE <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/algos/hyperparams/finetuned/mpe>`_
- `MaMujoco <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/algos/hyperparams/finetuned/mamujoco>`_

Simply add **--finetuned** when you run from the terminal command to use the finetuned hyper-parameters (if available).

Model Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Observation space varies with different environments, MARLlib automatically constructs the agent model to fit the diverse input shape, including:

- observation
- global state
- action mask
- additional information (e.g., minimap)

However, you can still customize your model in `model's config <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/models/configs>`_.
The supported architecture change includes:

- Observation/Global State Encoder: `CNN <https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/marl/models/configs/cnn_encoder.yaml>`_, `FC <https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/marl/models/configs/fc_encoder.yaml>`_
- `Recurrent Neural Network <https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/marl/models/configs/rnn.yaml>`_: GRU, LSTM
- `Q/Critic Value Mixer <https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/marl/models/configs/mixer.yaml>`_: VDN, QMIX

Running Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ray provides a robust multi-processing scheduling framework at the bottom of the MARLlib.
Together with RLlib, you can modify the `file of ray configuration <https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/marl/ray.yaml>`_ to adjust:

- sampling speed (worker number, CPU number)
- training speed (GPU acceleration)

and to switch

- running mode (local or distributed)
- parameter sharing strategy (all, group, individual)
- stop condition (iteration, reward, timestep)


Training
----------------------------------

MARLlib has two phases after you launch the whole process.

Phase 1:  Initialization

MARLlib initializes the environment and the agent model, producing a fake batch according to environment attributes and checking the sampling/training pipeline of the chosen algorithm.

Phase 2: Sampling & Training

Real jobs are assigned to workers and learner. MARL officially starts.

To start training, run:

.. code-block:: shell

    python marl/main.py --algo_config=$algo [--finetuned] --env-config=$env with env_args.map_name=$map


Examples

.. code-block:: shell

    python marl/main.py --algo_config=MAPPO --finetuned --env-config=smac with env_args.map_name=3m


Logging & Saving
----------------------------------

MARLlib uses the default logger provided by Ray in **ray.tune.CLIReporter**.
You can change the saved log location `here <https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/marl/algos/utils/log_dir_util.py>`_.


Develop & Debug mode
----------------------------------

Debug mode is designed for easier local debugging. To switch to debug mode, change the **local_mode** in **marl/ray.yaml** to True.
Debug mode will ignore the GPU settings and only use the CPU by default.
