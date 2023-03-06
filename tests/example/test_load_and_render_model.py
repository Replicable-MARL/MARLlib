from marllib import marl
import unittest

class TestPolicyRendering(unittest.TestCase):

    def test_policy_rendering(self):
        # prepare the environment
        env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)

        # initialize algorithm and load hyperparameters
        mappo = marl.algos.mappo(hyperparam_source="test")

        # build agent model based on env + algorithms + user preference if checked available
        model = marl.build_model(env, mappo, {"core_arch": "gru", "encode_layer": "128-256"})

        # rendering
        mappo.render(env, model,
                     restore_path={'params_path': "../../examples/checkpoint_000010/params.json",
                                   'model_path': "../../examples/checkpoint_000010/checkpoint-10"},
                     local_mode=True,
                     share_policy="all",
                     checkpoint_end=False)
