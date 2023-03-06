import unittest
from marllib import marl


class TestAlgo(unittest.TestCase):

    def test_collaborative_scenarios(self):
        for algo_name in dir(marl.algos):
            if algo_name[:2] != "__" and algo_name not in ["vdppo", "vda2c", "facmac", "qmix", "vdn", "happo", "hatrpo"]:
                if algo_name in ["ddpg", "maddpg", "facmac"]:
                    env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=False,
                                        continuous_actions=True)
                elif algo_name in ["iql"]:
                    continue
                else:
                    env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=False,
                                        continuous_actions=False)
                algo = getattr(marl.algos, algo_name)(hyperparam_source="test")
                model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "16-16"})
                algo.fit(env, model, stop={"training_iteration": 3}, local_mode=False, num_gpus=1,
                             num_workers=2, share_policy="all", checkpoint_end=False)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
