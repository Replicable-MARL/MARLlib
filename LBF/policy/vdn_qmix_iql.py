from gym.spaces import Dict as GymDict, Tuple, Box, Discrete
import sys
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.qmix.qmix import DEFAULT_CONFIG as QMIX_CONFIG, QMixTrainer
from ray.rllib.agents.trainer import Trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.concurrency_ops import Concurrently
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.replay_ops import Replay, StoreToReplayBuffer
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.execution.train_ops import TrainOneStep, UpdateTargetNetwork
from ray.rllib.utils.typing import TrainerConfigDict
from ray.util.iter import LocalIterator
from LBF.env.lbf_rllib_qmix import RllibLBF_QMIX
from LBF.util.qmix_tools import QMixReplayBuffer, QMixTorchPolicy_Customized
import sys


"""
This QMiX/VDN version is based on but different from that rllib built-in qmix_policy
1. the replay buffer is now standard localreplaybuffer instead of simplereplaybuffer
2. the QMIX loss is modified to be align with pymarl
3. provide reward standardize option
4. provide model optimizer option
5. follow DQN execution plan
"""


def run_vdn_qmix_iql(args, common_config, env_config, map_name, stop):
    if args.neural_arch not in ["GRU"]:
        print("{} arch not supported for QMIX/VDN".format(args.neural_arch))
        sys.exit()

    if not args.force_coop:
        print("competitive settings are not suitable for QMIX/VDN")
        sys.exit()

    single_env = RllibLBF_QMIX(env_config)
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    obs_space = Tuple([obs_space] * env_config["num_agents"])
    act_space = Tuple([act_space] * env_config["num_agents"])

    # align with LBF/env/lbf_rllib_qmix.py reset() function in line 41-50
    grouping = {
        "group_1": ["agent_{}".format(i) for i in range(env_config["num_agents"])],
    }

    # QMIX/VDN algo needs grouping env
    register_env(
        "grouped_lbf",
        lambda _: RllibLBF_QMIX(env_config).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space))

    mixer_dict = {
        "QMIX": "qmix",
        "VDN": "vdn",
        "IQL": None
    }

    # take care, when total sampled step > learning_starts, the training begins.
    # at this time, if the number of episode in buffer is less than train_batch_size,
    # then will cause dead loop where training never start.
    episode_limit = env_config["max_episode_steps"]
    train_batch_size = 4 if args.local_mode else 32
    buffer_slot = 100 if args.local_mode else 1000
    learning_starts = episode_limit * train_batch_size

    config = {
        "seed": common_config["seed"],
        "env": "grouped_lbf",
        "model": {
            "max_seq_len": episode_limit + 1,  # dynamic
        },
        "mixer": mixer_dict[args.run],
        "num_gpus": args.num_gpus,
    }

    QMIX_CONFIG.update(
        {
            "rollout_fragment_length": 1,
            "buffer_size": buffer_slot * episode_limit,  # in timesteps
            "train_batch_size": train_batch_size,  # in sequence
            "target_network_update_freq": episode_limit * 100,  # in timesteps
            "learning_starts": learning_starts,
            "lr": 0.0005,  # default
            "exploration_config": {
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.05,
                "epsilon_timesteps": 50000,  # Timesteps over which to anneal epsilon.
            },
            "evaluation_interval": args.evaluation_interval,
        })

    QMIX_CONFIG["reward_standardize"] = True  # this may affect the final performance so much if you turn off
    QMIX_CONFIG["training_intensity"] = None
    QMIX_CONFIG["optimizer"] = "RMSprop"  # or Adam

    def execution_plan_qmix(trainer: Trainer, workers: WorkerSet,
                            config: TrainerConfigDict, **kwargs) -> LocalIterator[dict]:
        # A copy of the DQN algorithm execution_plan.
        # Modified to be compatiable with QMIX.
        # Original QMIX replay bufferv(SimpleReplayBuffer) has bugs on replay() function
        # here we use QMixReplayBuffer inherited from LocalReplayBuffer

        local_replay_buffer = QMixReplayBuffer(
            learning_starts=config["learning_starts"],
            capacity=config["buffer_size"],
            replay_batch_size=config["train_batch_size"],
            replay_sequence_length=config.get("replay_sequence_length", 1),
            replay_burn_in=config.get("burn_in", 0),
            replay_zero_init_states=config.get("zero_init_states", True)
        )
        # Assign to Trainer, so we can store the LocalReplayBuffer's
        # data when we save checkpoints.
        trainer.local_replay_buffer = local_replay_buffer

        rollouts = ParallelRollouts(workers, mode="bulk_sync")

        # We execute the following steps concurrently:
        # (1) Generate rollouts and store them in our local replay buffer. Calling
        # next() on store_op drives this.
        store_op = rollouts.for_each(
            StoreToReplayBuffer(local_buffer=local_replay_buffer))

        def update_prio(item):
            samples, info_dict = item
            return info_dict

        # (2) Read and train on experiences from the replay buffer. Every batch
        # returned from the LocalReplay() iterator is passed to TrainOneStep to
        # take a SGD step, and then we decide whether to update the target network.
        post_fn = config.get("before_learn_on_batch") or (lambda b, *a: b)

        train_step_op = TrainOneStep(workers)

        replay_op = Replay(local_buffer=local_replay_buffer) \
            .for_each(lambda x: post_fn(x, workers, config)) \
            .for_each(train_step_op) \
            .for_each(update_prio) \
            .for_each(UpdateTargetNetwork(
            workers, config["target_network_update_freq"]))

        # Alternate deterministically between (1) and (2). Only return the output
        # of (2) since training metrics are not available until (2) runs.
        train_op = Concurrently(
            [store_op, replay_op],
            mode="round_robin",
            output_indexes=[1],
            round_robin_weights=[1, 1])

        return StandardMetricsReporting(train_op, workers, config)

    QMixTrainer_ = QMixTrainer.with_updates(
        name="QMIX",
        default_config=QMIX_CONFIG,
        default_policy=QMixTorchPolicy_Customized,
        get_policy_class=None,
        execution_plan=execution_plan_qmix)

    results = tune.run(QMixTrainer_,
                       name=args.run + "_" + args.neural_arch + "_" + map_name,
                       stop=stop,
                       config=config,
                       verbose=1)

    return results
