from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.tune.utils import merge_dicts
from SMAC.util.mappo_tools import *
from SMAC.util.vda2c_tools import *
from SMAC.util.vdppo_tools import *


def run_vdppo_sum_mix(args, common_config, env_config, stop):

    """
    for bug mentioned https://github.com/ray-project/ray/pull/20743
    make sure sgd_minibatch_size > max_seq_len
    """

    obs_shape = env_config["obs_shape"]
    n_ally = env_config["n_ally"]
    n_enemy = env_config["n_enemy"]
    state_shape = env_config["state_shape"]
    n_actions = env_config["n_actions"]
    rollout_fragment_length = env_config["rollout_fragment_length"]

    sgd_minibatch_size = 128
    while sgd_minibatch_size < rollout_fragment_length:
        sgd_minibatch_size *= 2

    config = {
        "env": "smac",
        "sgd_minibatch_size": sgd_minibatch_size,
        "num_sgd_iter": args.num_sgd_iter,
        "model": {
            "custom_model": "{}_ValueMixer".format(args.neural_arch),
            "max_seq_len": rollout_fragment_length,
            "custom_model_config": {
                "token_dim": args.token_dim,
                "ally_num": n_ally,
                "enemy_num": n_enemy,
                "self_obs_dim": obs_shape,
                "state_dim": state_shape,
                "mixer": "qmix" if args.run == "MIX-VDPPO" else "vdn",
                "mixer_emb_dim": 64,
            },
        },
    }
    config.update(common_config)

    VDPPO_CONFIG = merge_dicts(
        PPO_CONFIG,
        {
            "agent_num": n_ally,
            "state_dim": state_shape,
            "self_obs_dim": obs_shape,
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
