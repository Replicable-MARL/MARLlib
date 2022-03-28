from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

from ray.tune.utils import merge_dicts

from MaMujoco.util.mappo_tools import *
from MaMujoco.util.vda2c_tools import *
from MaMujoco.util.vdppo_tools import *


def run_vdppo_sum_mix(args, common_config, env_config, stop):

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
                "n_agents": env_config["ally_num"],
                "mixer": "qmix" if args.run == "MIX-VDPPO" else "vdn",
                "mixer_emb_dim": 64,
                "state_dim": env_config["state_dim"]
            },
        },
    }

    config.update(common_config)

    VDPPO_CONFIG = merge_dicts(
        PPO_CONFIG,
        {
            "agent_num": env_config["ally_num"],
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

    return results
