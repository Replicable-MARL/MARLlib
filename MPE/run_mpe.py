from ray.rllib.models import ModelCatalog
import sys
from ray.tune import register_env
from ray import tune

from ray.rllib.utils.test_utils import check_learning_achieved
from pettingzoo.mpe import simple_adversary_v2, simple_crypto_v2, simple_v2, simple_push_v2, simple_tag_v2, \
    simple_spread_v2, simple_reference_v2, simple_world_comm_v2, simple_speaker_listener_v3
from MPE.config_mpe import get_train_parser

from MPE.model.torch_gru import *
from MPE.model.torch_gru_cc import *
from MPE.model.torch_lstm import *
from MPE.model.torch_lstm_cc import *
from MPE.model.torch_vd_ppo_a2c_gru_lstm import *
from MPE.util.vda2c_tools import *
from MPE.util.vdppo_tools import *
from MPE.env.mpe_rllib import RllibMPE

from MPE.policy.pg_a2c_a3c import run_pg_a2c_a3c
from MPE.policy.vdn_qmix import run_vdn_qmix
from MPE.policy.ppo import run_ppo
from MPE.policy.vda2c import run_vda2c_sum_mix
from MPE.policy.vdppo import run_vdppo_sum_mix
from MPE.policy.maa2c import run_maa2c
from MPE.policy.mappo import run_mappo
from MPE.policy.ddpg import run_ddpg
from MPE.policy.maddpg import run_maddpg
from MPE.policy.r2d2 import run_r2d2
from MPE.policy.coma import run_coma

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


if __name__ == "__main__":
    args = get_train_parser().parse_args()
    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    ###################
    ### environment ###
    ###################

    if args.map == "simple_adversary":
        env = simple_adversary_v2.env(continuous_actions=args.continues)
    elif args.map == "simple_crypto":
        env = simple_crypto_v2.env(continuous_actions=args.continues)
    elif args.map == "simple":
        env = simple_v2.env(continuous_actions=args.continues)
    elif args.map == "simple_push":
        env = simple_push_v2.env(continuous_actions=args.continues)
    elif args.map == "simple_tag":
        env = simple_tag_v2.env(continuous_actions=args.continues)
    elif args.map == "simple_spread":
        env = simple_spread_v2.env(continuous_actions=args.continues)
    elif args.map == "simple_reference":
        env = simple_reference_v2.env(continuous_actions=args.continues)
    elif args.map == "simple_world_comm":
        env = simple_world_comm_v2.env(continuous_actions=args.continues)
    elif args.map == "simple_speaker_listener":
        env = simple_speaker_listener_v3.env(continuous_actions=args.continues)

    else:
        assert NotImplementedError
        print("Scenario {} not exists in pettingzoo".format(args.map))
        sys.exit()

    register_env(args.map, lambda _: RllibMPE(env))

    test_env = RllibMPE(env)
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    n_agents = test_env.num_agents
    test_env.close()

    env_config = {
        "n_agents": n_agents
    }

    #############
    ### model ###
    #############

    # Independent
    ModelCatalog.register_custom_model(
        "GRU_IndependentCritic", Torch_GRU_Model)
    ModelCatalog.register_custom_model(
        "LSTM_IndependentCritic", Torch_LSTM_Model)

    # CTDE(centralized critic)
    ModelCatalog.register_custom_model(
        "GRU_CentralizedCritic", Torch_GRU_CentralizedCritic_Model)
    ModelCatalog.register_custom_model(
        "LSTM_CentralizedCritic", Torch_LSTM_CentralizedCritic_Model)

    # Value Decomposition(mixer)
    ModelCatalog.register_custom_model("GRU_ValueMixer", Torch_GRU_Model_w_Mixer)
    ModelCatalog.register_custom_model("LSTM_ValueMixer", Torch_LSTM_Model_w_Mixer)

    ##############
    ### policy ###
    ##############

    if args.share_policy:
        policies = {"shared_policy"}
        policy_mapping_fn = (
            lambda agent_id, episode, **kwargs: "shared_policy")
    else:
        policies = {
            agent_name: (None, obs_space, act_space, {}) for agent_name in test_env.agents
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = lambda agent_id: agent_id

    policy_function_dict = {
        "PG": run_pg_a2c_a3c,
        "A2C": run_pg_a2c_a3c,
        "A3C": run_pg_a2c_a3c,
        "R2D2": run_r2d2,
        "VDN": run_vdn_qmix,
        "QMIX": run_vdn_qmix,
        "PPO": run_ppo,
        "MIX-VDA2C": run_vda2c_sum_mix,
        "SUM-VDA2C": run_vda2c_sum_mix,
        "MIX-VDPPO": run_vdppo_sum_mix,
        "SUM-VDPPO": run_vdppo_sum_mix,
        "MAA2C": run_maa2c,
        "MAPPO": run_mappo,
        "COMA": run_coma,
        "DDPG": run_ddpg,
        "MADDPG": run_maddpg
    }

    #####################
    ### common config ###
    #####################

    common_config = {
        "env": args.map,
        "num_gpus_per_worker": 0.2,
        "num_gpus": 0.6,
        "num_workers": 0,
        "train_batch_size": 1000,
        "rollout_fragment_length": 30,
        "horizon": 200,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn
        },
        # "callbacks": SmacCallbacks,
        "framework": args.framework,
    }

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    ##################
    ### run script ###
    ###################

    results = policy_function_dict[args.run](args, common_config, env_config, stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()

    ray.shutdown()
