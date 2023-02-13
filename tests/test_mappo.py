import unittest
import marl
from envs.base_env.mpe import REGISTRY as MPE_REGISTRY, policy_mapping_dict as mpe_policy_mapping_dict

'''
MAPPO test case
available scenario train 
one per iteration
'''

ENV = 'mpe'

class TestMPEEnv(unittest.TestCase):

    def test_all_scenarios_from_marllib_api(self):
        mappo = marl.algos.mappo(hyperparam_source=ENV)
        for scenario in MPE_REGISTRY.keys():
            print(scenario)
            env = marl.make_env(environment_name=ENV, map_name=scenario)
            mappo.fit(env, stop={'training_iteration': 5}, local_mode=True, num_gpus=0,
                      num_workers=1, share_policy='group', checkpoint_end=True)

    # def test_all_scenarios_from_pymarl_cmdline(self):
    #     mappo = marl.algos.mappo(hyperparam_source=ENV)
    #     for scenario in MPE_REGISTRY.keys():
    #         print(scenario)
    #         env = marl.make_env(environment_name=ENV, map_name=scenario)
    #         mappo.fit(env, stop={'training_iteration': 5}, local_mode=True, num_gpus=0,
    #                   num_workers=1, share_policy='group', checkpoint_end=True)
    #
    # def test_all_scenarios_from_vanilla_trainer(self):
    #     mappo = marl.algos.mappo(hyperparam_source=ENV)
    #     for scenario in MPE_REGISTRY.keys():
    #         print(scenario)
    #         env = marl.make_env(environment_name=ENV, map_name=scenario)
    #         mappo.fit(env, stop={'training_iteration': 5}, local_mode=True, num_gpus=0,
    #                   num_workers=1, share_policy='group', checkpoint_end=True)

    # def test_rendering_all_scenarios(self):
    #     mappo = marl.algos.mappo(hyperparam_source=ENV)
    #     for scenario in MPE_REGISTRY.keys():
    #         print(scenario)
    #         env = marl.make_env(environment_name=ENV, map_name=scenario)
    #         mappo.render(env, local_mode=True, num_gpus=0, num_workers=1, share_policy='group',
    #                      restore_path='../results/render/{}/mappo_{}_checkpoint_00005/checkpoint-5'.format(ENV, scenario),
    #                      checkpoint_end=False)



if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
