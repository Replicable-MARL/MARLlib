import unittest
from marllib import marl
import random

class TestMPEEnv(unittest.TestCase):

    def test_simple_case(self):
        algo_name_ls = []
        for algo_name in dir(marl.algos):
            if algo_name[:2] != "__":
                algo_name_ls.append(algo_name)

        test_num = 3
        algo_name_subset = random.sample(algo_name_ls, test_num)

        for algo_name in algo_name_subset:
            if algo_name in ["ddpg", "maddpg", "facmac"]:
                env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True,
                                    continuous_actions=True)
                algo = getattr(marl.algos, algo_name)(hyperparam_source="test")
                model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "64-64"})
                algo.fit(env, model, stop={"training_iteration": 3}, local_mode=False, num_gpus=0,
                             num_workers=2, share_policy="all", checkpoint_end=False)
            elif algo_name in ["happo", "hatrpo"]:
                continue
            else:
                env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
                algo = getattr(marl.algos, algo_name)(hyperparam_source="test")
                model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "64-64"})
                algo.fit(env, model, stop={"training_iteration": 3}, local_mode=False, num_gpus=0,
                             num_workers=2, share_policy="all", checkpoint_end=False)



if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
