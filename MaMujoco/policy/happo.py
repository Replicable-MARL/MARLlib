"""

Current now is same as MAPPO, need runnable firstly.

"""
from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

from MaMujoco.util.happo_tools import add_another_agent_and_gae
from MaMujoco.util.happo_tools import ppo_surrogate_loss

from MaMujoco.util.mappo_tools import setup_torch_mixins
from MaMujoco.util.mappo_tools import TorchLR
from MaMujoco.util.mappo_tools import TorchKLCoeffMixin
from MaMujoco.util.mappo_tools import TorchEntropyCoeffSchedule
from MaMujoco.util.mappo_tools import CentralizedValueMixin


def run_happo(args, common_config, env_config, stop):
    """
            for bug mentioned https://github.com/ray-project/ray/pull/20743
            make sure sgd_minibatch_size > max_seq_len
            """
    sgd_minibatch_size = 128
    while sgd_minibatch_size < args.horizon:
        sgd_minibatch_size *= 2

    config = {
        "env": args.map,
        "horizon": 1000,
        "num_sgd_iter": 5,
        "sgd_minibatch_size": sgd_minibatch_size,
        "lr": 5e-6,
        # "epoch": 5,
        "model": {
            "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
            "custom_model_config": {
                "agent_num": env_config["ally_num"],
                "state_dim": env_config["state_dim"]
            },
            "vf_share_layers": True
        },
    }
    config.update(common_config)

    HAPPOTorchPolicy = PPOTorchPolicy.with_updates(
        name="HAPPOTorchPolicy",
        get_default_config=lambda: PPO_CONFIG,
        postprocess_fn=add_another_agent_and_gae,
        loss_fn=ppo_surrogate_loss,
        before_init=setup_torch_mixins,
        mixins=[
            TorchLR, TorchEntropyCoeffSchedule, TorchKLCoeffMixin,
            CentralizedValueMixin
        ])

    def get_policy_class(config_):
        if config_["framework"] == "torch":
            return HAPPOTorchPolicy

    HAPPOTrainer = PPOTrainer.with_updates(
        name="HAPPOTrainer",
        default_policy=HAPPOTorchPolicy,
        get_policy_class=get_policy_class,
    )

    results = tune.run(HAPPOTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                       config=config,
                       verbose=1)

    return results
