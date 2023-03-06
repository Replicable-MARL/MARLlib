import unittest
from marllib import marl


class TestAlgo(unittest.TestCase):

    def test_cooperative_scenarios(self):
        for algo_name in dir(marl.algos):
            if algo_name[:2] != "__":
                if algo_name in ["ddpg", "maddpg", "facmac"]:
                    env = marl.make_env(environment_name="mamujoco", map_name="2AgentAnt")
                elif algo_name in ["happo", "hatrpo"]:
                    continue
                else:
                    env = marl.make_env(environment_name="smac", map_name="3m")
                one_algo = getattr(marl.algos, algo_name)(hyperparam_source="test")
                model = marl.build_model(env, one_algo, {"core_arch": "mlp", "encode_layer": "16-16"})
                one_algo.fit(env, model, stop={"training_iteration": 3}, local_mode=False, num_gpus=1,
                             num_workers=2, share_policy="all", checkpoint_end=False)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
