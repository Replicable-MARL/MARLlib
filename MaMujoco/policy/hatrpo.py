"""

Current now is same as MAPPO, need runnable firstly.

"""
from ray import tune
from ray.rllib.agents import trainer
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

from MaMujoco.util.happo_tools import add_another_agent_and_gae, make_happo_optimizers
from MaMujoco.util.happo_tools import surrogate_loss_for_ppo_and_trpo

from MaMujoco.util.mappo_tools import setup_torch_mixins
from MaMujoco.util.mappo_tools import TorchLR
from MaMujoco.util.mappo_tools import TorchKLCoeffMixin
from MaMujoco.util.mappo_tools import TorchEntropyCoeffSchedule
from MaMujoco.util.mappo_tools import CentralizedValueMixin
from ray.rllib.utils.torch_ops import apply_grad_clipping
from MaMujoco.util.happo_tools import grad_extra_for_trpo
import torch

from ray.rllib.agents.ppo import ppo


def run_hatrpo(args, common_config, env_config, stop):
    """
            for bug mentioned https://github.com/ray-project/ray/pull/20743
            make sure sgd_minibatch_size > max_seq_len
            """
    sgd_minibatch_size = 128
    while sgd_minibatch_size < args.horizon:
        sgd_minibatch_size *= 2

    config = {}
    config.update(common_config)

    config.update({
        "seed": 1,
        "env": args.map,
        "horizon": 1000,
        "num_sgd_iter": 5,  # ppo-epoch
        "train_batch_size": 4000,
        "sgd_minibatch_size": sgd_minibatch_size,
        # "lr": 5e-5,
        "grad_clip": 10,
        "clip_param": 0.3,  # ppo-clip
        "use_critic": True,
        # "critic_lr": 5e-3,
        "gamma": 0.99,
        "model": {
            "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
            "custom_model_config": {
                "agent_num": env_config["ally_num"],
                "state_dim": env_config["state_dim"],
                'normal_value': True
            },
            "vf_share_layers": True,
        },
    })

    PPO_CONFIG.update({
        'critic_lr': 1e-3,
        # 'actor_lr': 5e-5,
        'lr': 5e-6,
        "lr_schedule": [
            (0, 5e-6),
            (int(1e7), 1e-8),
        ]
    })

    HAPPOTorchPolicy = PPOTorchPolicy.with_updates(
        name="HAPPOTorchPolicy",
        get_default_config=lambda: PPO_CONFIG,
        postprocess_fn=add_another_agent_and_gae,
        loss_fn=surrogate_loss_for_ppo_and_trpo('TRPO'),
        before_init=setup_torch_mixins,
        # optimizer_fn=make_happo_optimizers,
        extra_grad_process_fn=grad_extra_for_trpo,
        mixins=[
            TorchEntropyCoeffSchedule, TorchKLCoeffMixin,
            CentralizedValueMixin, TorchLR
        ])

    def get_policy_class(config_):
        if config_["framework"] == "torch":
            return HAPPOTorchPolicy

    HAPPOTrainer = PPOTrainer.with_updates(
        name="#Paper-same-performance-after-use-logits-optimization",
        default_policy=HAPPOTorchPolicy,
        get_policy_class=get_policy_class,
    )

    results = tune.run(HAPPOTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                       config=config,
                       verbose=1)

    return results
