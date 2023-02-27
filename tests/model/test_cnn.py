import unittest
from marllib import marl


class TestMAgentEnv(unittest.TestCase):

    def test_cnn_encoder(self):
        for algo_name in dir(marl.algos):
            if algo_name[:2] != "__":
                if algo_name not in ["ddpg", "maddpg", "facmac", "happo", "hatrpo"]:
                    algo = getattr(marl.algos, algo_name)(hyperparam_source="test")
                    for core_arch in ["mlp", "gru"]:
                        env = marl.make_env(environment_name="football", map_name="academy_pass_and_shoot_with_keeper",
                                            force_coop=True)
                        model = marl.build_model(env, algo, {"core_arch": core_arch})
                        algo.fit(env, model, stop={'training_iteration': 3}, local_mode=False, num_gpus=1,
                                 num_workers=2, share_policy='all', checkpoint_end=False)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
