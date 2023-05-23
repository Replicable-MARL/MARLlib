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


class TestBaseOnMPE(unittest.TestCase):

    #################
    ### algorithm ###
    #################

    # HA algorithm
    def test_a1_hatrpo(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
        algo = marl.algos.hatrpo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="individual", checkpoint_end=False)

    def test_a2_happo(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
        algo = marl.algos.happo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="individual", checkpoint_end=False)

    # def test_a21_happo_global_state(self):
    #     env = marl.make_env(environment_name="smac", map_name="3m", force_coop=True)
    #     algo = marl.algos.happo(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="individual", checkpoint_end=False)
    #
    # CC algorithm
    def test_b1_maa2c(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_adversary")
        algo = marl.algos.maa2c(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "lstm", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="group", checkpoint_end=False)

    # def test_b12_maa2c_global_state(self):
    #     env = marl.make_env(environment_name="smac", map_name="3m", force_coop=True)
    #     algo = marl.algos.maa2c(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "lstm", "encode_layer": "8-8"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="group", checkpoint_end=False)
    #
    def test_b2_coma(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_adversary", continuous_actions=False)
        algo = marl.algos.coma(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "lstm", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="group", checkpoint_end=False)

    def test_b3_matrpo(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_adversary")
        algo = marl.algos.matrpo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "lstm", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="group", checkpoint_end=False)

    def test_b4_mappo(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread")
        algo = marl.algos.mappo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="individual", checkpoint_end=False)

    # def test_b5_maddpg_global_state(self):
    #     env = marl.make_env(environment_name="mamujoco", map_name="2AgentAnt", force_coop=True)
    #     algo = marl.algos.maddpg(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="all", checkpoint_end=False)

    def test_b51_maddpg(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", continuous_actions=True)
        algo = marl.algos.maddpg(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_b51_maddpg_smooth_target(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", continuous_actions=True)
        algo = marl.algos.maddpg(hyperparam_source="test", smooth_target_policy=True)
        model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    # IL algorithm
    def test_c1_ia2c(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread")
        algo = marl.algos.ia2c(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    # def test_c2_itrpo(self):
    #     env = marl.make_env(environment_name="smac", map_name="3m")
    #     algo = marl.algos.itrpo(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "lstm", "encode_layer": "8-8"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="group", checkpoint_end=False)

    def test_c3_ippo(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread")
        algo = marl.algos.ippo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "lstm", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="individual", checkpoint_end=False)

    def test_c3_iddpg(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", continuous_actions=True)
        algo = marl.algos.iddpg(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_c31_iddpg_smooth_target(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", continuous_actions=True)
        algo = marl.algos.iddpg(hyperparam_source="test", smooth_target_policy=True)
        model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_c4_iql(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=False)
        algo = marl.algos.iql(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    # VD algorithms

    def test_d1_qmix(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=False)
        algo = marl.algos.qmix(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_d2_vdn(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=False)
        algo = marl.algos.vdn(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    # def test_d21_vdn_smac_action_mask_and_state(self):
    #     env = marl.make_env(environment_name="smac", map_name="3m", force_coop=True)
    #     algo = marl.algos.vdn(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="all", checkpoint_end=False)

    def test_d23_vdn_adam(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
        algo = marl.algos.vdn(hyperparam_source="test", optimizer="adam")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_d3_vda2c(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
        algo = marl.algos.vda2c(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="group", checkpoint_end=False)

    # def test_d33_vda2c_global_state(self):
    #     env = marl.make_env(environment_name="smac", map_name="3m", force_coop=True)
    #     algo = marl.algos.vda2c(hyperparam_source="test")
    #     model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
    #     algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
    #              num_workers=2, share_policy="group", checkpoint_end=False)

    def test_d4_vdppo(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
        algo = marl.algos.vdppo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "lstm", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="individual", checkpoint_end=False)

    def test_d5_facmac(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
        algo = marl.algos.facmac(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_d51_facmac_smooth_target(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", continuous_actions=True)
        algo = marl.algos.facmac(hyperparam_source="test", smooth_target_policy=True)
        model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
