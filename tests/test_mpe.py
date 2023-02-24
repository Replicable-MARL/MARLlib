import unittest
from marllib import marl
from marllib.envs.base_env.mpe import REGISTRY as MPE_REGISTRY

'''
MAPPO test case
available scenario train 
one per iteration
'''


class TestAlgo(unittest.TestCase):

    def test_mappo_on_all_scenarios(self):
        mappo = marl.algos.mappo(hyperparam_source="mpe")
        for scenario in MPE_REGISTRY.keys():
            print(scenario)
            env = marl.make_env(environment_name="mpe", map_name=scenario)
            model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "64-64"})
            mappo.fit(env, model, stop={'training_iteration': 3}, local_mode=True, num_gpus=0,
                      num_workers=3, share_policy='group', checkpoint_end=False)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
