"""A simple example of setting up a multi-agent version of GFootball with rllib.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.models.catalog import ModelCatalog
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.a3c.a3c import DEFAULT_CONFIG as A3C_CONFIG
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG
from ray.tune.utils import merge_dicts
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG

from MaMuJoco.config_mamujoco import get_train_parser
from MaMuJoco.env.mamujoco_rllib import RllibMAMujoco
from MaMuJoco.util.mappo_tools import *
from MaMuJoco.util.maa2c_tools import *
from MaMuJoco.util.vda2c_tools import *
from MaMuJoco.util.vdppo_tools import *
from MaMuJoco.model.torch_gru import Torch_GRU_Model
from MaMuJoco.model.torch_lstm import Torch_LSTM_Model
from MaMuJoco.model.torch_gru_cc import Torch_GRU_CentralizedCritic_Model
from MaMuJoco.model.torch_lstm_cc import Torch_LSTM_CentralizedCritic_Model
from MaMuJoco.model.torch_vd_ppo_a2c_gru_lstm import Torch_LSTM_Model_w_Mixer, Torch_GRU_Model_w_Mixer

# from MaMuJoco.util.vdppo_tools import *

# from https://github.com/schroederdewitt/multiagent_mujoco
env_args_dict = {
    "2AgentAnt": {"scenario": "Ant-v2",
                  "agent_conf": "2x4",
                  "agent_obsk": 1,
                  "episode_limit": 1000},
    "2AgentAntDiag": {"scenario": "Ant-v2",
                      "agent_conf": "2x4d",
                      "agent_obsk": 1,
                      "episode_limit": 1000},
    "4AgentAnt": {"scenario": "Ant-v2",
                  "agent_conf": "4x2",
                  "agent_obsk": 1,
                  "episode_limit": 1000},
    "2AgentHalfCheetah": {"scenario": "HalfCheetah-v2",
                          "agent_conf": "2x3",
                          "agent_obsk": 1,
                          "episode_limit": 1000},
    "6AgentHalfCheetah": {"scenario": "HalfCheetah-v2",
                          "agent_conf": "6x1",
                          "agent_obsk": 1,
                          "episode_limit": 1000},
    "3AgentHopper": {"scenario": "Hopper-v2",
                     "agent_conf": "3x1",
                     "agent_obsk": 0,
                     "episode_limit": 1000},
    "2AgentHumanoid": {"scenario": "Humanoid-v2",
                       "agent_conf": "9|8",
                       "agent_obsk": 1,
                       "episode_limit": 1000},
    "2AgentHumanoidStandup": {"scenario": "HumanoidStandup-v2",
                              "agent_conf": "9|8",
                              "agent_obsk": 1,
                              "episode_limit": 1000},
    "2AgentReacher": {"scenario": "Reacher-v2",
                      "agent_conf": "2x1",
                      "agent_obsk": 1,
                      "episode_limit": 1000},
    "2AgentSwimmer": {"scenario": "Swimmer-v2",
                      "agent_conf": "2x1",
                      "agent_obsk": 1,
                      "episode_limit": 1000},
    "2AgentWalker": {"scenario": "Walker2d-v2",
                     "agent_conf": "2x3",
                     "agent_obsk": 1,
                     "episode_limit": 1000},
    "ManyagentSwimmer": {"scenario": "manyagent_swimmer",
                         "agent_conf": "10x2",
                         "agent_obsk": 1,
                         "episode_limit": 1000},
    "ManyagentAnt": {"scenario": "manyagent_ant",
                     "agent_conf": "2x3",
                     "agent_obsk": 1,
                     "episode_limit": 1000},
}

# TODO VDA2C VDPPO (only action)
if __name__ == "__main__":
    args = get_train_parser().parse_args()
    ray.init(local_mode=True)

    env_config = env_args_dict[args.map]

    register_env(args.map, lambda _: RllibMAMujoco(env_config))

    # Independent
    ModelCatalog.register_custom_model("LSTM", Torch_LSTM_Model)
    ModelCatalog.register_custom_model("GRU", Torch_GRU_Model)

    # CTDE(centralized critic (only action))
    ModelCatalog.register_custom_model(
        "GRU_CentralizedCritic", Torch_GRU_CentralizedCritic_Model)
    ModelCatalog.register_custom_model(
        "LSTM_CentralizedCritic", Torch_LSTM_CentralizedCritic_Model)

    # Value Decomposition(mixer)
    ModelCatalog.register_custom_model("GRU_ValueMixer", Torch_GRU_Model_w_Mixer)
    ModelCatalog.register_custom_model("LSTM_ValueMixer", Torch_LSTM_Model_w_Mixer)
    # ModelCatalog.register_custom_model("CNN_UPDeT_ValueMixer", Torch_CNN_Transformer_Model_w_Mixer)

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    single_env = RllibMAMujoco(env_config)
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    state_dim = single_env.state_dim
    ally_num = single_env.num_agents

    policies = {
        "policy_{}".format(i): (None, obs_space, act_space, {}) for i in range(ally_num)
    }
    policy_ids = list(policies.keys())

    common_config = {
        "num_gpus_per_worker": args.num_gpus_per_worker,
        "train_batch_size": 1000,
        "num_workers": args.num_workers,
        "num_gpus": args.num_gpus,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": tune.function(
                lambda agent_id: policy_ids[int(agent_id[6:])]),
        },
        "framework": args.framework,
    }

    if args.run in ["PG", "A2C", "A3C", "R2D2"]:
        config = {
            "env": args.map,
            "horizon": args.horizon,
            "model": {
                "custom_model": args.neural_arch,
            },
        }

        config.update(common_config)
        results = tune.run(args.run,
                           name=args.run + "_" + args.neural_arch + "_" + args.map,
                           stop=stop,
                           config=config,
                           verbose=1)

    elif args.run in ["SUM-VDA2C", "MIX-VDA2C"]:


        config = {
            "env": args.map,
            "horizon": args.horizon,
            "model": {
                "custom_model": "{}_ValueMixer".format(args.neural_arch),
                "custom_model_config": {
                    "n_agents": ally_num,
                    "mixer": "qmix" if args.run == "MIX-VDA2C" else "vdn",
                    "mixer_emb_dim": 64,
                    "state_dim": state_dim
                },
            },
        }
        config.update(common_config)

        VDA2C_CONFIG = merge_dicts(
            A2C_CONFIG,
            {
                "agent_num": ally_num,
            }
        )

        VDA2CTFPolicy = A3CTFPolicy.with_updates(
            name="VDA2CTFPolicy",
            postprocess_fn=value_mix_centralized_critic_postprocessing,
            loss_fn=value_mix_actor_critic_loss,
            grad_stats_fn=central_vf_stats_a2c, )

        VDA2CTorchPolicy = A3CTorchPolicy.with_updates(
            name="VDA2CTorchPolicy",
            get_default_config=lambda: VDA2C_CONFIG,
            postprocess_fn=value_mix_centralized_critic_postprocessing,
            loss_fn=value_mix_actor_critic_loss,
            mixins=[ValueNetworkMixin, MixingValueMixin],
        )


        def get_policy_class(config_):
            if config_["framework"] == "torch":
                return VDA2CTorchPolicy


        VDA2CTrainer = A2CTrainer.with_updates(
            name="VDA2CTrainer",
            default_policy=VDA2CTFPolicy,
            get_policy_class=get_policy_class,
        )

        results = tune.run(VDA2CTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                           config=config, verbose=1)

    elif args.run == "MAA2C":  # centralized A2C

        config = {
            "env": args.map,
            "horizon": args.horizon,
            "model": {
                "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                "custom_model_config": {
                    "agent_num": ally_num,
                    "state_dim": state_dim
                }
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
                           name=args.run + "_" + args.neural_arch + "_" + args.map,
                           stop=stop,
                           config=config,
                           verbose=1)

    elif args.run in ["PPO", "APPO"]:

        """
        for bug mentioned https://github.com/ray-project/ray/pull/20743
        make sure sgd_minibatch_size > max_seq_len
        """
        sgd_minibatch_size = 128
        while sgd_minibatch_size < args.horizon:
            sgd_minibatch_size *= 2

        config = {
            "env": args.map,
            "horizon": args.horizon,
            "num_sgd_iter": 5,
            "sgd_minibatch_size": sgd_minibatch_size,
            "model": {
                "custom_model": args.neural_arch,
            },
        }
        config.update(common_config)
        results = tune.run(args.run, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop, config=config,
                           verbose=1)

    elif args.run in ["SUM-VDPPO", "MIX-VDPPO"]:

        """
        for bug mentioned https://github.com/ray-project/ray/pull/20743
        make sure sgd_minibatch_size > max_seq_len
        """
        sgd_minibatch_size = 128
        while sgd_minibatch_size < args.horizon:
            sgd_minibatch_size *= 2

        config = {
            "env": args.map,
            "horizon": args.horizon,
            "num_sgd_iter": 5,
            "sgd_minibatch_size": sgd_minibatch_size,
            "model": {
                "custom_model": "{}_ValueMixer".format(args.neural_arch),
                "custom_model_config": {
                    "n_agents": ally_num,
                    "mixer": "qmix" if args.run == "MIX-VDPPO" else "vdn",
                    "mixer_emb_dim": 64,
                    "state_dim": state_dim
                },
            },
        }

        config.update(common_config)

        VDPPO_CONFIG = merge_dicts(
            PPO_CONFIG,
            {
                "agent_num": ally_num,
            }
        )

        # not used
        VDPPOTFPolicy = PPOTFPolicy.with_updates(
            name="VDPPOTFPolicy",
            postprocess_fn=value_mix_centralized_critic_postprocessing,
            loss_fn=value_mix_ppo_surrogate_loss,
            before_loss_init=setup_tf_mixins,
            grad_stats_fn=central_vf_stats_ppo,
            mixins=[
                LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
                ValueNetworkMixin, MixingValueMixin
            ])

        VDPPOTorchPolicy = PPOTorchPolicy.with_updates(
            name="VDPPOTorchPolicy",
            get_default_config=lambda: VDPPO_CONFIG,
            postprocess_fn=value_mix_centralized_critic_postprocessing,
            loss_fn=value_mix_ppo_surrogate_loss,
            before_init=setup_torch_mixins,
            mixins=[
                TorchLR, TorchEntropyCoeffSchedule, TorchKLCoeffMixin,
                ValueNetworkMixin, MixingValueMixin
            ])

        def get_policy_class(config_):
            if config_["framework"] == "torch":
                return VDPPOTorchPolicy

        VDPPOTrainer = PPOTrainer.with_updates(
            name="VDPPOTrainer",
            default_policy=VDPPOTFPolicy,
            get_policy_class=get_policy_class,
        )

        results = tune.run(VDPPOTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                           config=config,
                           verbose=1)


    elif args.run == "MAPPO":  # centralized PPO

        """
        for bug mentioned https://github.com/ray-project/ray/pull/20743
        make sure sgd_minibatch_size > max_seq_len
        """
        sgd_minibatch_size = 128
        while sgd_minibatch_size < args.horizon:
            sgd_minibatch_size *= 2

        config = {
            "env": args.map,
            "horizon": args.horizon,
            "num_sgd_iter": 5,
            "sgd_minibatch_size": sgd_minibatch_size,
            "model": {
                "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
                "custom_model_config": {
                    "agent_num": ally_num,
                    "state_dim": state_dim
                }
            },
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

        results = tune.run(MAPPOTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                           config=config,
                           verbose=1)

    ray.shutdown()
