"""

Current now is same as MAPPO, need runnable firstly.

"""
from ray import tune
from ray.rllib.agents import trainer
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

from SMAC.util.happo_tools import add_another_agent_and_gae, make_happo_optimizers
from SMAC.util.happo_tools import ppo_surrogate_loss

from SMAC.util.mappo_tools import setup_torch_mixins
from SMAC.util.mappo_tools import TorchLR
from SMAC.util.mappo_tools import TorchKLCoeffMixin
from SMAC.util.mappo_tools import TorchEntropyCoeffSchedule
from SMAC.util.mappo_tools import CentralizedValueMixin
from ray.rllib.utils.torch_ops import apply_grad_clipping
import torch

from ray.rllib.agents.ppo import ppo


def run_happo(args, common_config, env_config, stop):
    """
            for bug mentioned https://github.com/ray-project/ray/pull/20743
            make sure sgd_minibatch_size > max_seq_len
            """
    rollout_fragment_length = env_config["rollout_fragment_length"]

    sgd_minibatch_size = 128
    while sgd_minibatch_size < rollout_fragment_length:
        sgd_minibatch_size *= 2

    config = {}
    config.update(common_config)

    obs_shape = env_config["obs_shape"]
    n_ally = env_config["n_ally"]
    n_enemy = env_config["n_enemy"]
    state_shape = env_config["state_shape"]

    config.update({
        "seed": 1,
        "env": "smac",
        "horizon": 160,
        "num_sgd_iter": 5,  # ppo-epoch
        "train_batch_size": 3200,
        "sgd_minibatch_size": sgd_minibatch_size,
        # "lr": 5e-5,
        "grad_clip": 10,
        "clip_param": 0.3,  # ppo-clip
        "use_critic": True,
        # "critic_lr": 5e-3,
        "gamma": 0.99,
        "model": {
            "custom_model": "{}_CentralizedCritic".format(args.neural_arch),
            "max_seq_len": rollout_fragment_length,
            "custom_model_config": {
                "token_dim": args.token_dim,
                "ally_num": n_ally,
                "enemy_num": n_enemy,
                "self_obs_dim": obs_shape,
                "state_dim": state_shape,
                'normal_value': True
            },
            "vf_share_layers": True,
        },
    })

    PPO_CONFIG.update({
        'critic_lr': 5e-4,
        # 'actor_lr': 5e-5,
        'lr': 5e-4,
        "lr_schedule": [
            (0, 5e-4),
            (int(1e7), 1e-8),
        ],
        "agent_num": n_ally,
        "state_dim": state_shape,
        "self_obs_dim": obs_shape,
        "centralized_critic_obs_dim": -1,
    })

    HAPPOTorchPolicy = PPOTorchPolicy.with_updates(
        name="HAPPOTorchPolicy",
        get_default_config=lambda: PPO_CONFIG,
        postprocess_fn=add_another_agent_and_gae,
        loss_fn=ppo_surrogate_loss,
        before_init=setup_torch_mixins,
        # optimizer_fn=make_happo_optimizers,
        extra_grad_process_fn=apply_grad_clipping,
        mixins=[
            TorchEntropyCoeffSchedule, TorchKLCoeffMixin,
            CentralizedValueMixin, TorchLR
        ])

    def get_policy_class(config_):
        if config_["framework"] == "torch":
            return HAPPOTorchPolicy

    HAPPOTrainer = PPOTrainer.with_updates(
        name="test-happo-in-smac",
        default_policy=HAPPOTorchPolicy,
        get_policy_class=get_policy_class,
    )

    results = tune.run(HAPPOTrainer, name=args.run + "_" + args.neural_arch + "_" + args.map, stop=stop,
                       config=config,
                       verbose=1)

    return results
