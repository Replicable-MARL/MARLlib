import unittest
from marllib import marl


class TestSimpleCase(unittest.TestCase):

    def test_simple_case(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
        mappo = marl.algos.mappo(hyperparam_source="test")
        model = marl.build_model(env, mappo, {"core_arch": "gru", "encode_layer": "64-64"})
        mappo.fit(env, model,
                  stop={"training_iteration": 3},
                  local_mode=False,
                  num_gpus=0,
                  num_workers=5,
                  share_policy='all',
                  checkpoint_freq=10)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
