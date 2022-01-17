from ray.rllib.env import PettingZooEnv
from ray import tune
from ray.tune import register_env
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.a3c.a3c import DEFAULT_CONFIG as A3C_CONFIG

from ray.rllib.agents.dqn.r2d2 import DEFAULT_CONFIG, R2D2Trainer
from ray.rllib.agents.dqn.r2d2_tf_policy import R2D2TFPolicy

from pettingzoo.classic import hanabi_v4

from config_hanabi import *
from Hanabi.model.torch_mask_gru import *
from Hanabi.model.torch_mask_gru_cc import *
from Hanabi.model.torch_mask_lstm import *
from Hanabi.model.torch_mask_lstm_cc import *
from Hanabi.model.torch_mask_r2d2 import *

from Hanabi.utils.mappo_tools import *
from Hanabi.utils.maa2c_tools import *

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

if __name__ == '__main__':
    args = get_train_parser().parse_args()

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    # Independent
    ModelCatalog.register_custom_model(
        "GRU_IndependentCritic", Torch_ActionMask_GRU_Model)
    ModelCatalog.register_custom_model(
        "LSTM_IndependentCritic", Torch_ActionMask_LSTM_Model)

    # CTDE(centralized critic)
    ModelCatalog.register_custom_model(
        "GRU_CentralizedCritic", Torch_ActionMask_GRU_CentralizedCritic_Model)
    ModelCatalog.register_custom_model(
        "LSTM_CentralizedCritic", Torch_ActionMask_LSTM_CentralizedCritic_Model)

    agent_num = args.num_players
    env = hanabi_v4.env(players=agent_num)

    register_env("Hanabi", lambda _: PettingZooEnv(env))

    test_env = PettingZooEnv(env)
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    n_agents = len(test_env.agents)

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    common_config = {
        "env": "Hanabi",
        "num_gpus_per_worker": args.num_gpus_per_worker,
        "num_gpus": args.num_gpus,
        "num_workers": args.num_workers,
        "train_batch_size": 1000,
        "rollout_fragment_length": 30,
        "horizon": 200,
        "multiagent": {
            "policies": {
                agent_name: (None, obs_space, act_space, {}) for agent_name in test_env.agents
            },
            "policy_mapping_fn": lambda agent_id: agent_id
        },
        # "callbacks": SmacCallbacks,
        "framework": args.framework,
    }

    if args.run in ["QMIX", "VDN"]:

        print("Hanabi is a turn based game, QMIX/VDN is not suitable")
        raise ValueError()

    elif args.run in ["R2D2"]:  # similar to IQL in recurrent/POMDP mode

        config = {
            "model": {
                "custom_model": "{}_IndependentCritic".format(args.neural_arch),
            },
            "framework": args.framework,
        }
        config.update(common_config)


        def get_policy_class(config_):
            if config_["framework"] == "torch":
                return R2D2WithMaskPolicy


        DEFAULT_CONFIG['dueling'] = False  # with mask, only support no dueling arch, default use dueling arch
        R2D2Trainer_ = R2D2Trainer.with_updates(
            name="R2D2_Trainer",
            default_config=DEFAULT_CONFIG,
            default_policy=R2D2TFPolicy,
            get_policy_class=get_policy_class,
        )

        results = tune.run(R2D2Trainer_, name=args.run + "_" + args.neural_arch + "_" + "Hanabi", stop=stop,
                           config=config,
                           verbose=1)

    elif args.run in ["PG", "A2C", "A3C"]:  # PG need define action mask GRU / only torch now

        config = {
            "model": {
                "custom_model": "{}_IndependentCritic".format(args.neural_arch),
            },
        }

        config.update(common_config)

        tune.run(
            args.run,
            name=args.run + "_" + args.neural_arch + "_" + "Hanabi",
            stop=stop,
            config=config,
            verbose=1
        )

    elif args.run == "MAA2C":  # centralized A2C

        config = {
            "model": {
                "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                "custom_model_config": {
                    "agent_num": n_agents
                },
            },
        }
        config.update(common_config)

        MAA2CTFPolicy = A3CTFPolicy.with_updates(
            name="MAA2CTFPolicy",
            postprocess_fn=centralized_critic_postprocessing,
            loss_fn=loss_with_central_critic_a2c,
            grad_stats_fn=central_vf_stats_a2c,
            mixins=[
                CentralizedValueMixin
            ])

        MAA2CTorchPolicy = A3CTorchPolicy.with_updates(
            name="MAA2CTorchPolicy",
            get_default_config=lambda: A3C_CONFIG,
            postprocess_fn=centralized_critic_postprocessing,
            loss_fn=loss_with_central_critic_a2c,
            mixins=[
                CentralizedValueMixin
            ])


        def get_policy_class(config_):
            if config_["framework"] == "torch":
                return MAA2CTorchPolicy


        MAA2CTrainer = A2CTrainer.with_updates(
            name="MAA2CTrainer",
            default_policy=MAA2CTFPolicy,
            get_policy_class=get_policy_class,
        )

        results = tune.run(MAA2CTrainer,
                           name=args.run + "_" + args.neural_arch + "_" + "Hanabi",
                           stop=stop,
                           config=config,
                           verbose=1)

    elif args.run in ["PPO"]:
        config = {
            "model": {
                "custom_model": "{}_IndependentCritic".format(args.neural_arch),
            },
            "num_sgd_iter": 5,
        }

        config.update(common_config)

        tune.run(
            args.run,
            name=args.run + "_" + args.neural_arch + "_" + "Hanabi",
            stop=stop,
            config=config,
            verbose=1
        )

    elif args.run in ["MAPPO"]:

        config = {
            "model": {
                "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                "custom_model_config": {
                    "agent_num": n_agents
                }
            },
            "num_sgd_iter": 5,
        }
        config.update(common_config)

        MAPPOTFPolicy = PPOTFPolicy.with_updates(
            name="MAPPOTFPolicy",
            postprocess_fn=centralized_critic_postprocessing,
            loss_fn=loss_with_central_critic,
            before_loss_init=setup_tf_mixins,
            grad_stats_fn=central_vf_stats_ppo,
            mixins=[
                LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
                CentralizedValueMixin
            ])

        MAPPOTorchPolicy = PPOTorchPolicy.with_updates(
            name="MAPPOTorchPolicy",
            get_default_config=lambda: PPO_CONFIG,
            postprocess_fn=centralized_critic_postprocessing,
            loss_fn=loss_with_central_critic,
            before_init=setup_torch_mixins,
            mixins=[
                TorchLR, TorchEntropyCoeffSchedule, TorchKLCoeffMixin,
                CentralizedValueMixin
            ])


        def get_policy_class(config_):
            if config_["framework"] == "torch":
                return MAPPOTorchPolicy


        MAPPOTrainer = PPOTrainer.with_updates(
            name="MAPPOTrainer",
            default_policy=MAPPOTFPolicy,
            get_policy_class=get_policy_class,
        )

        results = tune.run(MAPPOTrainer,
                           name=args.run + "_" + args.neural_arch + "_" + "Hanabi",
                           stop=stop,
                           config=config,
                           verbose=1)

    ray.shutdown()
