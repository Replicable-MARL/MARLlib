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

import unittest
from marllib import marl


class TestAlgo(unittest.TestCase):

    def test_cooperative_scenarios(self):
        for algo_name in dir(marl.algos):
            if "_" not in algo_name:
                if algo_name in ["iddpg", "maddpg", "facmac"]:
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
