import unittest
from marllib import marl
from marllib.envs.base_env.mpe import REGISTRY as MPE_REGISTRY

'''
MAPPO test case
available scenario train 
one per iteration
'''


class TestMPEEnv(unittest.TestCase):

    def test_all_algorithms_on_some_scenario(self):
        marl.algos.__dict__.items()
        for algo in dir(marl.algos):
            if algo[:2] != "__":
                if algo in ["ddpg", "maddpg", "facmac"]:
                    env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True,
                                        continuous_actions=True)
                    one_algo = getattr(marl.algos, algo)(hyperparam_source="common")
                    model = marl.build_model(env, one_algo, {"core_arch": "mlp", "encode_layer": "64-64"})
                    one_algo.fit(env, model, stop={'training_iteration': 3}, local_mode=False, num_gpus=0,
                                 num_workers=3, share_policy='all', checkpoint_end=False)
                elif algo in ["happo", "hatrpo"]:
                    pass
                else:
                    env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
                    one_algo = getattr(marl.algos, algo)(hyperparam_source="common")
                    model = marl.build_model(env, one_algo, {"core_arch": "mlp", "encode_layer": "64-64"})
                    one_algo.fit(env, model, stop={'training_iteration': 3}, local_mode=False, num_gpus=0,
                                 num_workers=3, share_policy='all', checkpoint_end=False)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
