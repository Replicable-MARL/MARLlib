from ray import tune
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.a3c.a3c import DEFAULT_CONFIG as A3C_CONFIG
from ray.tune.utils import merge_dicts
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG
from Pommerman.util.vda2c_tools import *
from Pommerman.util.maa2c_tools import *
import sys

def run_vda2c_sum_mix(args, common_config, env_config, agent_list, stop):
    if args.neural_arch not in ["CNN_GRU", "CNN_LSTM"]:
        print("{} arch not supported for QMIX/VDN".format(args.neural_arch))
        sys.exit()

    if "Team" not in args.map:
        print("VDA2C is only for cooperative scenarios")
        sys.exit()

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
        "model": {
            "custom_model": "{}_ValueMixer".format(args.neural_arch),
            "custom_model_config": {
                "map_size": 11 if "One" not in args.map else 8,
                "agent_num": 2 if env_config["neural_agent_pos"] in [[0, 1], [2, 3]] else 4,
                "mixer": "qmix" if args.run == "MIX-VDA2C" else "vdn",
                "mixer_emb_dim": 64,
            },
        },
    }
    config.update(common_config)

    VDA2CTFPolicy = A3CTFPolicy.with_updates(
        name="VDA2CTFPolicy",
        postprocess_fn=value_mix_centralized_critic_postprocessing,
        loss_fn=value_mix_actor_critic_loss,
        grad_stats_fn=central_vf_stats_a2c,
        mixins=[
            CentralizedValueMixin
        ])

    A3C_CONFIG["grouping"] = grouping

    VDA2CTorchPolicy = A3CTorchPolicy.with_updates(
        name="VDA2CTorchPolicy",
        get_default_config=lambda: A3C_CONFIG,
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

    results = tune.run(VDA2CTrainer,
                       name=args.run + "_" + args.neural_arch + "_" + args.map,
                       stop=stop,
                       config=config,
                       verbose=1)

    return results


