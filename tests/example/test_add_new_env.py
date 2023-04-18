# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from examples.add_new_env import RLlibMAGym
import unittest


class TestAddEnv(unittest.TestCase):

    def test_add_env(self):
        # register new env
        ENV_REGISTRY["magym"] = RLlibMAGym
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
