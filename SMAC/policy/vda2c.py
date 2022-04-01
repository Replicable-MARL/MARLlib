from ray import tune
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.tune.utils import merge_dicts
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG
from SMAC.util.vda2c_tools import *
from SMAC.util.maa2c_tools import *

def run_vda2c_sum_mix(args, common_config, env_config, stop):

    obs_shape = env_config["obs_shape"]
    n_ally = env_config["n_ally"]
    n_enemy = env_config["n_enemy"]
    state_shape = env_config["state_shape"]
    n_actions = env_config["n_actions"]
    episode_limit = env_config["episode_limit"]

    episode_num = 10
    train_batch_size = episode_num * episode_limit

    config = {
        "env": "smac",
        "batch_mode": "complete_episodes",
        "train_batch_size": train_batch_size,
        "model": {
            "custom_model": "{}_ValueMixer".format(args.neural_arch),
            "max_seq_len": episode_limit,
            "custom_model_config": {
                "token_dim": args.token_dim,
                "ally_num": n_ally,
                "enemy_num": n_enemy,
                "self_obs_dim": obs_shape,
                "state_dim": state_shape,
                "mixer": "qmix" if args.run == "MIX-VDA2C" else "vdn",
                "mixer_emb_dim": 64,
            },
        },
    }
    config.update(common_config)

    VDA2C_CONFIG = merge_dicts(
        A2C_CONFIG,
        {
            "agent_num": n_ally,
            "state_dim": state_shape,
            "self_obs_dim": obs_shape,
            "rollout_fragment_length":episode_limit
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

    return results


