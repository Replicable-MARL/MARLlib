from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from Pommerman.util.mappo_tools import *
from Pommerman.util.vda2c_tools import *
from Pommerman.util.vdppo_tools import *


def run_vdppo_sum_mix(args, common_config, env_config, agent_list, stop):

    """
    for bug mentioned https://github.com/ray-project/ray/pull/20743
    make sure sgd_minibatch_size > max_seq_len
    """

    if args.neural_arch not in ["CNN_GRU", "CNN_LSTM"]:
        print("{} arch not supported for QMIX/VDN".format(args.neural_arch))
        raise ValueError()

    if "Team" not in args.map:
        print("VDA2C is only for cooperative scenarios")
        raise ValueError()

    if env_config["neural_agent_pos"] == [0, 1, 2, 3]:
        # 2 vs 2
        grouping = {
            "group_1": ["agent_{}".format(i) for i in [0, 1]],
            "group_2": ["agent_{}".format(i) for i in [2, 3]],
        }

    elif env_config["neural_agent_pos"] in [[0, 1], [2, 3]]:
        grouping = {
            "group_1": ["agent_{}".format(i) for i in [0, 1]],
        }

    else:
        print("Wrong agent position setting")
        raise ValueError

    config = {
        "env": "pommerman",
        "num_sgd_iter": 5,
        "model": {
            "custom_model": "{}_ValueMixer".format(args.neural_arch),
            "custom_model_config": {
                "map_size": 11 if "One" not in args.map else 8,
                "agent_num": 2 if env_config["neural_agent_pos"] in [[0, 1], [2, 3]] else 4,
                "mixer": "qmix" if args.run == "MIX-VDPPO" else "vdn",
                "mixer_emb_dim": 64,
            },
        },
    }
    config.update(common_config)

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

    PPO_CONFIG["grouping"] = grouping


    VDPPOTorchPolicy = PPOTorchPolicy.with_updates(
        name="VDPPOTorchPolicy",
        get_default_config=lambda: PPO_CONFIG,
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

    return results
