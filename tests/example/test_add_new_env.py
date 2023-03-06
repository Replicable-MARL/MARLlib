from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from examples.add_new_env import RllibMAGym
import unittest


class TestAddEnv(unittest.TestCase):

    def test_add_env(self):
        # register new env
        ENV_REGISTRY["magym"] = RllibMAGym
        # initialize env
        env = marl.make_env(environment_name="magym", map_name="Checkers")
        # pick mappo algorithms
        mappo = marl.algos.mappo(hyperparam_source="test")
        # customize model
        model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})
        # start learning
        mappo.fit(env, model,
                  stop={"training_iteration": 3},
                  local_mode=True,
                  num_gpus=1,
                  num_workers=2,
                  share_policy='all',
                  checkpoint_freq=50)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
