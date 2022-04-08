from gym.spaces import Dict as GymDict
from ray import tune
from ray.tune import register_env
from SMAC.model.torch_mask_updet_cc import *
from SMAC.util.r2d2_tools import *
from SMAC.model.torch_vdn_qmix_iql_model import *
from ray.rllib.agents.qmix.qmix import DEFAULT_CONFIG as QMIX_CONFIG
from ray.rllib.agents.trainer import Trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.concurrency_ops import Concurrently
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.replay_buffer import LocalReplayBuffer
from ray.rllib.execution.replay_ops import Replay, StoreToReplayBuffer
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.execution.train_ops import TrainOneStep, UpdateTargetNetwork, \
    MultiGPUTrainOneStep
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import TrainerConfigDict
from ray.util.iter import LocalIterator
from SMAC.env.starcraft2_rllib import StarCraft2Env_Rllib as SMAC
from SMAC.util.qmix_tools import QMixTorchPolicy_Customized, QMixReplayBuffer


def run_vdn_qmix_iql(args, common_config, env_config, stop):
    if args.neural_arch not in ["GRU", "UPDeT"]:
        assert NotImplementedError

    obs_shape = env_config["obs_shape"]
    n_ally = env_config["n_ally"]
    n_enemy = env_config["n_enemy"]
    state_shape = env_config["state_shape"]
    n_actions = env_config["n_actions"]
    episode_limit = env_config["episode_limit"]

    grouping = {
        "group_1": ["agent_{}".format(i) for i in range(n_ally)],
    }
    ## obs state setting here
    obs_space = Tuple([
                          GymDict({
                              "obs": Box(-2.0, 2.0, shape=(obs_shape,)),
                              "state": Box(-2.0, 2.0, shape=(state_shape,)),
                              "action_mask": Box(0.0, 1.0, shape=(n_actions,))
                          })] * n_ally
                      )
    act_space = Tuple([
                          Discrete(n_actions)
                      ] * n_ally)

    # QMIX/VDN need grouping
    register_env(
        "grouped_smac",
        lambda config: SMAC(config).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space))

    mixer_dict = {
        "QMIX": "qmix",
        "VDN": "vdn",
        "IQL": None
    }
    config = {
        "seed": common_config["seed"],
        "env": "grouped_smac",
        "env_config": {
            "map_name": args.map,
        },
        "model": {
            "max_seq_len": episode_limit + 1,  # dynamic
            "custom_model_config": {
                "neural_arch": args.neural_arch,
                "token_dim": args.token_dim,
                "ally_num": n_ally,
                "enemy_num": n_enemy,
                "self_obs_dim": obs_shape,
                "state_dim": state_shape
            },
        },
        "mixer": mixer_dict[args.run],
        "callbacks": common_config["callbacks"],
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": args.num_gpus,
        # "_disable_preprocessor_api": True
    }

    learning_starts = episode_limit * 32
    train_batch_size = 32
    QMIX_CONFIG.update(
        {
            "rollout_fragment_length": 1,
            "buffer_size": 5000 * episode_limit // 2,  # in timesteps
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
    QMIX_CONFIG["optimizer"] = "epymarl"  # pyamrl for RMSProp or epymarl for Adam

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

    results = tune.run(QMixTrainer_, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                       config=config, verbose=1)

    return results
