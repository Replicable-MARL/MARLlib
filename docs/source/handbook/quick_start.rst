.. _quick-start:

Quick Start
===========

.. contents::
    :local:
    :depth: 2

If you have not installed MARLlib yet, please refer to :ref:`basic-installation` before running.

Before Running
-----------------

.. figure:: ../images/configurations.png
    :align: center

    Prepare all the configuration files to start your MARL journey

MARLlib adopt a configuration based customization method to control the whole learning pipeline.
There are four configuration files you need to ensure correctness for your training demand.

- scenario: specify your environment/task settings
- algorithm: finetune your algorithm hyperparameters
- model: customize the model architecture
- ray/rllib: changing the basic training settings

We introduce them one by one.

Scenario Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MARLlib provides ten environments for you to conduct your experiment.
After installing :ref:`basic-installation`, you don't have to install all of these environments.
Simply follow the instruction :ref:`env` to install the environment you need and change the corresponding configuration.

All the scenario configurations are in  `env configure <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/envs/base_env/config>`_.

For instance, the Multiple Particle Environments (MPE) are set to accept only discrete action.
To allow continuous action, simply change **continuous_actions** in `mpe.yaml <https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/envs/base_env/config/mpe.yaml>`_ to **True**


Algorithm Hyper-parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After the environment is all set, you need to visit the MARL algorithms' hyper-parameter directory.
Each algorithm has different hyper-parameters to finetune with.

Most of the algorithms are sensitive to the environment settings.
This means you need to give a set of hyper-parameters that fit for current MARL task.

We provide a `commonly used hyper-parameters directory <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/algos/hyperparams/common>`_.
And a finetuned hyper-parameters sets for the four most used MARL environments/benchmarks, including

- `GRF  <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/algos/hyperparams/finetuned/football>`_
- `SMAC <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/algos/hyperparams/finetuned/smac>`_
- `MPE <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/algos/hyperparams/finetuned/mpe>`_
- `MAMuJoCo <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/algos/hyperparams/finetuned/mamujoco>`_

Simply add **--finetuned** when you run from the terminal command to use the finetuned hyper-parameters (if available).

Model Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Observation space varies with different environments, MARLlib automatically constructs the agent model to fit the diverse input shape, including:

- observation
- global state
- action mask
- additional information (e.g., minimap)

However, we leave space for you to customize your model in `model's config <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/models/configs>`_.
The supported architecture change includes:

- Observation/Global State Encoder: `CNN <https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/marl/models/configs/cnn_encoder.yaml>`_, `FC <https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/marl/models/configs/fc_encoder.yaml>`_
- `Recurrent Neural Network <https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/marl/models/configs/rnn.yaml>`_: GRU, LSTM
- `Q/Critic Value Mixer <https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/marl/models/configs/mixer.yaml>`_: VDN, QMIX

Running Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ray/RLlib provides a flexible multi-processing scheduling mechanism for MARLlib.
You can modify the `file of ray configuration <https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/marl/ray.yaml>`_ to adjust:

- sampling speed (worker number, CPU number)
- training speed (GPU acceleration)

and switch

- running mode (locally or distributed)
- parameter sharing strategy (all, group, individual)
- stop condition (iteration, reward, timestep)


Training
----------------------------------

MARLlib has two phases after you launch the whole process.

Phase 1:  Initialization

MARLlib initializes the environment and the agent model, producing a fake batch according to environment attributes and checking the sampling/training pipeline of the chosen algorithm.

Phase 2: Sampling & Training

Real jobs are assigned to workers and learner. Learning starts.

To start training, make sure you are under MARLlib directory and run:

.. code-block:: shell

    python marl/main.py --algo_config=$algo [--finetuned] --env_config=$env with env_args.map_name=$map

Available algorithms (case sensitive):

- iql
- pg
- a2c
- ddpg
- trpo
- ppo
- maa2c
- coma
- maddpg
- matrpo
- mappo
- hatrpo
- happo
- vdn
- qmix
- facmac
- vda2c
- vdppo

Available env-map pairs (case sensitive):

- smac: `smac maps <https://github.com/oxwhirl/smac/blob/master/smac/env/starcraft2/maps/smac_maps.py>`_
- mpe: `mpe map <https://github.com/Replicable-MARL/MARLlib/blob/main/envs/base_env/mpe.py>`_
- mamujoco: `mamujoco map <https://github.com/Replicable-MARL/MARLlib/blob/main/envs/base_env/mamujoco.py>`_
- football: `football map <https://github.com/Replicable-MARL/MARLlib/blob/main/envs/base_env/mamujoco.py>`_
- magent: `magent map <https://github.com/Replicable-MARL/MARLlib/blob/main/envs/base_env/magent.py>`_
- lbf: use `lbf config <https://github.com/Replicable-MARL/MARLlib/blob/main/envs/base_env/config/lbf.yaml>`_ to generate the map. Details can be found https://github.com/semitable/lb-foraging#usage
- rware: use `rware config <https://github.com/Replicable-MARL/MARLlib/blob/main/envs/base_env/config/rware.yaml>`_ to generate the map. Details can be found https://github.com/semitable/robotic-warehouse#naming-scheme
- pommerman: OneVsOne-v0, PommeFFACompetition-v0, PommeTeamCompetition-v0
- metadrive: Bottleneck, ParkingLot, Intersection, Roundabout, Tollgate
- hanabi: Hanabi-Very-Small, Hanabi-Full, Hanabi-Full-Minimal, Hanabi-Small

--finetuned is optional, force using the finetuned hyperparameter if available in `this directory <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/algos/hyperparams/finetuned>`_


Example on SMAC:

.. code-block:: shell

    python marl/main.py --algo_config=mappo --finetuned --env_config=smac with env_args.map_name=3m




Logging & Saving
----------------------------------

MARLlib uses the default logger provided by Ray in **ray.tune.CLIReporter**.
You can change the saved log location `here <https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/marl/algos/utils/log_dir_util.py>`_.


Develop & Debug mode
----------------------------------

Debug mode is designed for easier local debugging. To switch to debug mode, change the **local_mode** in **marl/ray.yaml** to True.
Debug mode will ignore the GPU settings and only use the CPU by default.
