import unittest
from marllib import marl

class TestMPEEnv(unittest.TestCase):

    def test_all_algorithms(self):
        for algo_name in dir(marl.algos):
            if "_" not in algo_name:
                if algo_name in ["ddpg", "maddpg", "facmac"]:
                    env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True,
                                        continuous_actions=True)
                    algo = getattr(marl.algos, algo_name)(hyperparam_source="test")
                    model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "16-16"})
                    algo.fit(env, model, stop={"training_iteration": 3}, local_mode=True, num_gpus=0,
                                 num_workers=0, share_policy="all", checkpoint_end=False)
                if algo_name in ["happo", "hatrpo"]:
                    env = marl.make_env(environment_name="mpe", map_name="simple_spread")
                    algo = getattr(marl.algos, algo_name)(hyperparam_source="test")
                    model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "16-16"})
                    algo.fit(env, model, stop={"training_iteration": 3}, local_mode=True, num_gpus=0,
                                 num_workers=2, share_policy="individual", checkpoint_end=False)
                else:
                    env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
                    algo = getattr(marl.algos, algo_name)(hyperparam_source="test")
                    model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "16-16"})
                    algo.fit(env, model, stop={"training_iteration": 3}, local_mode=False, num_gpus=0,
                                 num_workers=2, share_policy="all", checkpoint_end=False)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
