"""
example on how to wrap env to customize the group policy sharing
a.k.a
how to group as you like
"""
from examples.customize_policy_sharing import *
import unittest


class TestCustomizePolicySharing(unittest.TestCase):

    def test_customize_policy_sharing(self):
        # register new env
        ENV_REGISTRY["two_teams_smac"] = Two_Teams_SMAC
        # initialize env
        env = marl.make_env(environment_name="two_teams_smac", map_name="3m")
        # pick mappo algorithms
        mappo = marl.algos.mappo(hyperparam_source="test")
        # customize model
        model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})
        # start learning
        mappo.fit(env, model, stop={"training_iteration": 3}, local_mode=True,
                  num_gpus=1,
                  num_workers=2, share_policy='group', checkpoint_freq=50)


if __name__ == '__main__':
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
