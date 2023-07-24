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

    #############
    ### model ###
    #############

    def test_a01_hatrpo_hacc_mlp(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
        algo = marl.algos.hatrpo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="individual", checkpoint_end=False)

    def test_a02_hatrpo_hacc_gru(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
        algo = marl.algos.hatrpo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="individual", checkpoint_end=False)
    
    def test_a11_happo_hacc_mlp(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
        algo = marl.algos.happo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="individual", checkpoint_end=False)

    def test_a12_happo_hacc_gru(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
        algo = marl.algos.happo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="individual", checkpoint_end=False)

    def test_a13_happo_hacc_lstm(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
        algo = marl.algos.happo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "lstm", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="individual", checkpoint_end=False)

    def test_b11_facmac_ddpg_series_mlp(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
        algo = marl.algos.facmac(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_b12_facmac_ddpg_series_gru(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
        algo = marl.algos.facmac(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_b13_facmac_vdnmixer_mlp(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
        algo = marl.algos.facmac(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8", "mixer_arch": "vdn"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_b14_facmac_vdnmixer_gru(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
        algo = marl.algos.facmac(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8", "mixer_arch": "vdn"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_b11_maddpg_ddpg_series_mlp(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
        algo = marl.algos.maddpg(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_b12_maddpg_ddpg_series_gru(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
        algo = marl.algos.maddpg(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_b13_maddpg_ddpg_series_lstm(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
        algo = marl.algos.maddpg(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "lstm", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_c11_vdppo_vd_mlp(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
        algo = marl.algos.vdppo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="group", checkpoint_end=False)

    def test_c12_vdppo_vd_gru(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
        algo = marl.algos.vdppo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_c13_vdppo_vdn_mixer_mlp(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
        algo = marl.algos.vdppo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8", "mixer_arch": "vdn"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="group", checkpoint_end=False)

    def test_c14_vdppo_vdn_mixer_gru(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
        algo = marl.algos.vdppo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8", "mixer_arch": "vdn"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_d14_qmix_mixer_mlp(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
        algo = marl.algos.qmix(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_d12_qmix_mixer_gru(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
        algo = marl.algos.qmix(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_d11_vdn_mixer_mlp(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
        algo = marl.algos.vdn(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_d12_vdn_mixer_gru(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
        algo = marl.algos.vdn(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_e11_ippo_cnn(self):
        env = marl.make_env(environment_name="magent", map_name="battlefield")
        algo = marl.algos.ippo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "mlp"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="group", checkpoint_end=False)
    
    def test_e12_ippo_cnn(self):
        env = marl.make_env(environment_name="magent", map_name="battlefield")
        algo = marl.algos.ippo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="group", checkpoint_end=False)
    
    def test_f11_mappo_cnn(self):
        env = marl.make_env(environment_name="magent", map_name="battlefield")
        algo = marl.algos.mappo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "mlp"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="group", checkpoint_end=False)
    
    def test_f12_mappo_cnn(self):
        env = marl.make_env(environment_name="magent", map_name="battlefield")
        algo = marl.algos.mappo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="group", checkpoint_end=False)
    
    def test_f11_mappo_1d_encoder(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread")
        algo = marl.algos.mappo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "mlp"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="group", checkpoint_end=False)

    def test_f12_mappo_1d_encoder(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread")
        algo = marl.algos.mappo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="group", checkpoint_end=False)

    def test_g11_coma_mlp(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread")
        algo = marl.algos.coma(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "mlp"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="group", checkpoint_end=False)

    def test_g12_coma_gru(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread")
        algo = marl.algos.coma(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="group", checkpoint_end=False)


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
    
    def test_d3_vda2c(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
        algo = marl.algos.vda2c(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)
    
    def test_d4_vdppo(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
        algo = marl.algos.vdppo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "lstm", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)
    
    def test_d5_facmac(self):
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True, continuous_actions=True)
        algo = marl.algos.facmac(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="all", checkpoint_end=False)

    def test_all_algorithms(self):
    
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
        algo = marl.algos.happo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "lstm", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="individual", checkpoint_end=False)
    
        env = marl.make_env(environment_name="mpe", map_name="simple_spread")
        algo = marl.algos.hatrpo(hyperparam_source="test")
        model = marl.build_model(env, algo, {"core_arch": "lstm", "encode_layer": "8-8"})
        algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=0,
                 num_workers=2, share_policy="individual", checkpoint_end=False)
    
    
    
        for algo_name in dir(marl.algos):
            if "_" not in algo_name:
                if algo_name in ["iddpg", "maddpg", "facmac"]:
                    env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True,
                                        continuous_actions=True)
                    algo = getattr(marl.algos, algo_name)(hyperparam_source="test")
                    model = marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "8-8"})
                    algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=1,
                                 num_workers=2, share_policy="all", checkpoint_end=False)
                elif algo_name in ["happo", "hatrpo"]:
                    continue
                else:
                    env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
                    algo = getattr(marl.algos, algo_name)(hyperparam_source="test")
                    model = marl.build_model(env, algo, {"core_arch": "mlp", "encode_layer": "8-8"})
                    algo.fit(env, model, stop={"training_iteration": 1}, local_mode=True, num_gpus=1,
                                 num_workers=2, share_policy="group", checkpoint_end=False)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
