.. _quick-start:

Quick Start
===========

If you have not installed MARLlib yet, please refer to :ref:`installation` before running. We give two cases of running `Policy Space Response Oracle (PSRO) <https://arxiv.org/pdf/1711.00832.pdf>`_ to solve `Leduc Holdem <https://en.wikipedia.org/wiki/Texas_hold_%27em>`_, and `MADDPG <https://arxiv.org/abs/1706.02275>`_ to solve a particle multi-agent cooperative task, `Simple Spread <https://www.pettingzoo.ml/mpe/simple_spread>`_.


PSRO Learning
-------------

**Policy Space Response Oracle (PSRO)** is a population-based MARL algorithm which cooperates game-theory and MARL algorithm to solve multi-agent tasks in the scope of meta-game. At each iteration, the algorithm will generate some policy combinations and executes independent learning for each agent. Such a nested learning process comprises rollout, training, evaluation in sequence, and works circularly until the algorithm finds the estimated Nash Equilibrium. 

.. note:: If you want to use alpha\-rank to estimate the equilibrium, you need to install open\-spiel before that. Follow the :ref:`installation` to get more details.

Specify the environment
^^^^^^^^^^^^^^^^^^^^^^^

The first thing to start your training task is to determine the environment for policy learning. Here, we use the Leduc Hodlem environment as an example. If you want to use custom environment.

.. code-block:: python

    from marllib.envs.poker import poker_aec_env as leduc_holdem

    env = leduc_holdem.env(scenario_configs={"fixed_player": True}, env_id="leduc_poker")
    env_description = {
        "creator": leduc_holdem.env,
        "config": {
            "scenario_configs": {"fixed_player": True},
            "env_id": "leduc_poker",
        },
        "possible_agents": env.possible_agents,
        "observation_spaces": env.observation_spaces,
        "action_spaces": env.action_spaces
    }


Specify the training interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We need to determine which interface to training the PSRO algorithm. In this case, we use ``independent`` interface that trains single-agent reinforcement learning algorithm.

.. code-block:: python

    training = {
        "interface": {
            "type": "independent",
            "observation_spaces": observation_spaces,
            "action_spaces": action_spaces,
            "use_init_policy_pool": True,
        },
        "config": {
            "batch_size": args.batch_size,
            "update_interval": args.num_epoch,
        },
    },


Specify the rollout interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Specify the rollout interface in MARLlib is very simple, we've implemented several rollout interfaces to meet different distributed computing requirements. In this case, we use ``AsyncRollout`` to support the high-throughput data collection.

.. code-block:: python

    from marllib.rollout import rollout_func

    rollout = {
        "type": "async",
        "stopper": "simple_rollout",
        "stopper_config": {"max_step": 1000},
        "metric_type": "simple",
        "fragment_length": args.fragment_length,
        "num_episodes": args.num_episode,
        "num_env_per_worker": args.episode_seg,
        "max_step": 10,
        "postprocessor_types": ["copy_next_frame"],
    }


Specify the underlying (MA)RL algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PSRO requires an underlying RL algorithm to find the best response at each learning iteration, you need to specify the algorithm you want to use in this learning. As a standard implementation, the underlying algorithm is DQN.

.. code-block:: python

    algorithms = {
        "PSRO_DQN": {
            "name": "DQN",
            "custom_config": {
                "gamma": 1.0,
                "eps_min": 0,
                "eps_max": 1.0,
                "eps_anneal_time": 100,
                "lr": 1e-2,
            },
        }
    },


The completed distributed execution example is presented below.

.. code-block:: python

    """PSRO with DQN for Leduc Holdem"""

    from marllib.envs.poker import poker_aec_env as leduc_holdem
    from marllib.runner import run
    from marllib.rollout import rollout_func


    env = leduc_holdem.env(fixed_player=True)

    run(
        group="psro",
        name="leduc_poker",
        env_description=env_description,
        training=training,
        algorithms=algorithms,
        rollout=rollout,
        evaluation={
            "max_episode_length": 100,
            "num_episode": args.num_simulation,
        },  # dict(num_simulation=num_simulation, sim_max_episode_length=5),
        global_evaluator={
            "name": "psro",
            "config": {
                "stop_metrics": {"max_iteration": 1000, "loss_threshold": 2.0},
            },
        },
        dataset_config={"episode_capacity": args.buffer_size},
        task_mode="gt",  # gt: for pb-marl; marl: for the training of typical marl algorithms like maddpg
    )


Multi-agent Reinforcement Learning
----------------------------------

Similar to the above example. Users can run a multi-agent algorithm training on MARLlib by specificying environment, training and rollout configuration, also the algorithm used. The following example loads the configuration from an existing yaml file. For more details, please refer to the files under the examples directory.

.. code-block:: python

    import yaml
    import os

    from marllib.envs import MPE
    from marllib.runner import run

    with open(os.path.join(BASE_DIR, "examples/configs/mpe/maddpg_push_ball_nips.yaml"), "r") as f:
        config = yaml.safe_load(f)

    env_desc = config["env_description"]
    env_desc["config"] = env_desc.get("config", {})
    # load creator
    env_desc["creator"] = MPE
    env = MPE(**env_desc["config"])

    possible_agents = env.possible_agents
    observation_spaces = env.observation_spaces
    action_spaces = env.action_spaces

    env_desc["possible_agents"] = env.possible_agents
    env.close()
    env_desc["observation_spaces"] = env.observation_spaces
    env_desc["action_spaces"] = env.action_spaces

    training_config = config["training"]
    rollout_config = config["rollout"]

    training_config["interface"]["observation_spaces"] = observation_spaces
    training_config["interface"]["action_spaces"] = action_spaces

    run(
        group=config["group"],
        name=config["name"],
        env_description=env_desc,
        agent_mapping_func=lambda agent: "share",
        training=training_config,
        algorithms=config["algorithms"],
        # rollout configuration for each learned policy model
        rollout=rollout_config,
        evaluation=config.get("evaluation", {}),
        global_evaluator=config["global_evaluator"],
        dataset_config=config.get("dataset_config", {}),
        parameter_server=config.get("parameter_server", {}),
        use_init_policy_pool=False,
        task_mode="marl",
    )