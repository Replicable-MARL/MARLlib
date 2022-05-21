from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, ValueNetworkMixin, KLCoeffMixin
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
import re


def setup_torch_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTorchPolicy  (w/o ValueNetworkMixin).
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def get_policy_class(ppo_config, default_policy):
    def _get_policy_class(config_):
        if config_["framework"] == "torch":
            return default_policy(ppo_config)
    return _get_policy_class()


def _algos_var(config, key):

    str_fmt = config['algo_args'][key].strip()

    float_express = (r'\d*\.\d*', float)
    sci_float = (r'\d+\.?\d*e-\d+|\d+\.\d*e\d+', float)
    sci_int = (r'\d+e\d+', lambda n: int(float(n)))
    bool_express = (r'True|False', lambda s: s == 'True')
    int_express = (r'\d+', int)

    express_matches = [
        float_express, sci_float, sci_int, bool_express, int_express
    ]

    value = str_fmt

    for pat, type_f in express_matches:
        if re.search(pat, str_fmt):
            value = type_f(str_fmt)
            break

    return value


if __name__ == '__main__':
    assert _algos_var({'algo_args': {'test': 'True'}}, 'test') is True
    assert _algos_var({'algo_args': {'test': 'False'}}, 'test') is False
    assert _algos_var({'algo_args': {'test': '1e5'}}, 'test') == 100000
    assert _algos_var({'algo_args': {'test': '1e-5'}}, 'test') == 0.00001
    assert _algos_var({'algo_args': {'test': '1e0'}}, 'test') == 1
    assert _algos_var({'algo_args': {'test': '2e0'}}, 'test') == 2
    assert _algos_var({'algo_args': {'test': '1.01'}}, 'test') == 1.01
    assert _algos_var({'algo_args': {'test': '123'}}, 'test') == 123

    print('test done!')





