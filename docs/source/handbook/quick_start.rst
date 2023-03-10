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

We provide a `commonly used hyper-parameters directory <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/algos/hyperparams/common>`_,
a `test-only hyper-parameters directory <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/algos/hyperparams/test>`_, and
And a finetuned hyper-parameters sets for the three most used MARL environments/benchmarks, including

- `SMAC <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/algos/hyperparams/finetuned/smac>`_
- `MPE <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/algos/hyperparams/finetuned/mpe>`_
- `MAMuJoCo <https://github.com/Replicable-MARL/MARLlib/tree/sy_dev/marl/algos/hyperparams/finetuned/mamujoco>`_

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
- `Multiple Layers perceptron <https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/marl/models/configs/mlp.yaml>`_: MLP
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

.. code-block:: python

    from marllib import marl
    # prepare env
    env = marl.make_env(environment_name="mpe", map_name="simple_spread")
    # initialize algorithm with appointed hyper-parameters
    mappo = marl.algos.mappo(hyperparam_source="mpe")
    # build agent model based on env + algorithms + user preference
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})
    # start training
    mappo.fit(env, model, stop={"timesteps_total": 1000000}, checkpoint_freq=100, share_policy="group")

prepare the ``environment``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - task mode
     - api example
   * - cooperative
     - ``marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)``
   * - collaborative
     - ``marl.make_env(environment_name="mpe", map_name="simple_spread")``
   * - competitive
     - ``marl.make_env(environment_name="mpe", map_name="simple_adversary")``
   * - mixed
     - ``marl.make_env(environment_name="mpe", map_name="simple_crypto")``


Most of the popular environments in MARL research are supported by MARLlib:

.. list-table::
   :header-rows: 1

   * - Env Name
     - Learning Mode
     - Observability
     - Action Space
     - Observations
   * - :ref:`LBF`
     - cooperative + collaborative
     - Both
     - Discrete
     - 1D
   * - :ref:`RWARE`
     - cooperative
     - Partial
     - Discrete
     - 1D
   * - :ref:`MPE`
     - cooperative + collaborative + mixed
     - Both
     - Both
     - 1D
   * - :ref:`SMAC`
     - cooperative
     - Partial
     - Discrete
     - 1D
   * - :ref:`MetaDrive`
     - collaborative
     - Partial
     - Continuous
     - 1D
   * - :ref:`MAgent`
     - collaborative + mixed
     - Partial
     - Discrete
     - 2D
   * - :ref:`Pommerman`
     - collaborative + competitive + mixed
     - Both
     - Discrete
     - 2D
   * - :ref:`MAMuJoCo`
     - cooperative
     - Partial
     - Continuous
     - 1D
   * - :ref:`Football`
     - collaborative + mixed
     - Full
     - Discrete
     - 2D
   * - :ref:`Hanabi`
     - cooperative
     - Partial
     - Discrete
     - 1D


Each environment has a readme file, standing as the instruction for this task, including env settings, installation,
and important notes.

initialize the  ``algorithm``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - running target
     - api example
   * - train & finetune
     - ``marl.algos.mappo(hyperparam_source=$ENV)``
   * - develop & debug
     - ``marl.algos.mappo(hyperparam_source="test")``
   * - 3rd party env
     - ``marl.algos.mappo(hyperparam_source="common")``


Here is a chart describing the characteristics of each algorithm:

.. list-table::
   :header-rows: 1

   * - algorithm
     - support task mode
     - discrete action
     - continuous action
     - policy type
   * - :ref:`IQL`
     - all four
     - :heavy_check_mark:
     -
     - off-policy
   * - :ref:`IPG`
     - all four
     - :heavy_check_mark:
     - :heavy_check_mark:
     - on-policy
   * - :ref:`IA2C`
     - all four
     - :heavy_check_mark:
     - :heavy_check_mark:
     - on-policy
   * - :ref:`IDDPG`
     - all four
     -
     - :heavy_check_mark:
     - off-policy
   * - :ref:`ITRPO`
     - all four
     - :heavy_check_mark:
     - :heavy_check_mark:
     - on-policy
   * - :ref:`IPPO`
     - all four
     - :heavy_check_mark:
     - :heavy_check_mark:
     - on-policy
   * - :ref:`COMA`
     - all four
     - :heavy_check_mark:
     -
     - on-policy
   * - :ref:`MADDPG`
     - all four
     -
     - :heavy_check_mark:
     - off-policy
   * - :ref:`MAA2C`
     - all four
     - :heavy_check_mark:
     - :heavy_check_mark:
     - on-policy
   * - :ref:`MATRPO`
     - all four
     - :heavy_check_mark:
     - :heavy_check_mark:
     - on-policy
   * - :ref:`MAPPO`
     - all four
     - :heavy_check_mark:
     - :heavy_check_mark:
     - on-policy
   * - :ref:`HATRPO`
     - cooperative
     - :heavy_check_mark:
     - :heavy_check_mark:
     - on-policy
   * - :ref:`HAPPO`
     - cooperative
     - :heavy_check_mark:
     - :heavy_check_mark:
     - on-policy
   * - :ref:`VDN`
     - cooperative
     - :heavy_check_mark:
     -
     - off-policy
   * - :ref:`QMIX`
     - cooperative
     - :heavy_check_mark:
     -
     - off-policy
   * - :ref:`FACMAC`
     - cooperative
     -
     - :heavy_check_mark:
     - off-policy
   * - :ref:`VDA2C`
     - cooperative
     - :heavy_check_mark:
     - :heavy_check_mark:
     - on-policy
   * - :ref:`VDPPO`
     - cooperative
     - :heavy_check_mark:
     - :heavy_check_mark:
     - on-policy

***all four**\ : cooperative collaborative competitive mixed

construct the agent  ``model``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - model arch
     - api example
   * - MLP
     - ``marl.build_model(env, algo, {"core_arch": "mlp")``
   * - GRU
     - ``marl.build_model(env, algo, {"core_arch": "gru"})``
   * - LSTM
     - ``marl.build_model(env, algo, {"core_arch": "lstm"})``
   * - encoder arch
     - ``marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "128-256"})``


kick off the training ``algo.fit``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - setting
     - api example
   * - train
     - ``algo.fit(env, model)``
   * - debug
     - ``algo.fit(env, model, local_mode=True)``
   * - stop condition
     - ``algo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000000})``
   * - policy sharing
     - ``algo.fit(env, model, share_policy='all') # or 'group' / 'individual'``
   * - save model
     - ``algo.fit(env, model, checkpoint_freq=100, checkpoint_end=True)``
   * - GPU accelerate
     - ``algo.fit(env, model, local_mode=False, num_gpus=1)``
   * - CPU accelerate
     - ``algo.fit(env, model, local_mode=False, num_workers=5)``


policy inference ``algo.render``

.. list-table::
   :header-rows: 1

   * - setting
     - api example
   * - render
     - ``algo.render(env, model, local_mode=True, restore_path='path_to_model')``


By default, all the models will be saved at ``/home/username/ray_results/experiment_name/checkpoint_xxxx``

Logging & Saving
----------------------------------

MARLlib uses the default logger provided by Ray in **ray.tune.CLIReporter**.
You can change the saved log location `here <https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/marl/algos/utils/log_dir_util.py>`_.

