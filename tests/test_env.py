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
import ray


class TestBaseOnMAPPO(unittest.TestCase):

    ###################
    ### environment ###
    ###################

    def test_a1_mpe(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread")
        algo = marl.algos.mappo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="group", checkpoint_end=False)

    # def test_b1_magent(self):
    #     env = marl.make_env(environment_name="magent", map_name="gather")
    #     algo = marl.algos.mappo(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "mlp"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="group", checkpoint_end=False)
    #
    # def test_c1_smac(self):
    #     env = marl.make_env(environment_name="smac", map_name="3m")
    #     algo = marl.algos.mappo(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="group", checkpoint_end=False)
    #
    # def test_d1_grf(self):
    #     env = marl.make_env(environment_name="football", map_name="academy_pass_and_shoot_with_keeper")
    #     algo = marl.algos.mappo(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "mlp"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="group", checkpoint_end=False)

    # def test_e1_rware(self):
    #     env = marl.make_env(environment_name="rware", map_name="rware")
    #     algo = marl.algos.mappo(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "mlp","encode_layer": "8-8"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="group", checkpoint_end=False)
    #
    # def test_f1_lbf(self):
    #     env = marl.make_env(environment_name="lbf", map_name="lbf")
    #     algo = marl.algos.mappo(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="group", checkpoint_end=False)
    #
    # def test_g1_pommerman(self):
    #     env = marl.make_env(environment_name="pommerman", map_name="PommeTeamCompetition-v0")
    #     algo = marl.algos.mappo(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="group", checkpoint_end=False)
    #
    # def test_h1_metadrive(self):
    #     env = marl.make_env(environment_name="metadrive", map_name="Bottleneck")
    #     algo = marl.algos.mappo(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=False, num_gpus=1,
    #              num_workers=3, share_policy="group", checkpoint_end=False)
    #
    # def test_i1_hanabi(self):
    #     env = marl.make_env(environment_name="hanabi", map_name="Hanabi-Very-Small")
    #     algo = marl.algos.mappo(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="group", checkpoint_end=False)
    #
    # def test_j1_mate(self):
    #     env = marl.make_env(environment_name="mate", map_name="MATE-4v2-9-v0")
    #     algo = marl.algos.mappo(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="group", checkpoint_end=False)
    #
    # def test_k1_gobigger(self):
    #     env = marl.make_env(environment_name="gobigger", map_name="st_t1p2")
    #     algo = marl.algos.mappo(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="group", checkpoint_end=False)
    #
    # def test_l1_overcooked(self):
    #     env = marl.make_env(environment_name="overcooked", map_name="asymmetric_advantages")
    #     algo = marl.algos.mappo(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="group", checkpoint_end=False)
    #
    # def test_m1_voltage(self):
    #     env = marl.make_env(environment_name="voltage", map_name="case33_3min_final")
    #     algo = marl.algos.mappo(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="group", checkpoint_end=False)
    #
    # def test_n1_mamujoco(self):
    #     env = marl.make_env(environment_name="mamujoco", map_name="2AgentAnt")
    #     algo = marl.algos.mappo(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="group", checkpoint_end=False)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
