from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, ValueNetworkMixin, KLCoeffMixin
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
import re


def get_agent_num(policy):
    custom_config = policy.config["model"]["custom_model_config"]
    n_agents = custom_config["num_agents"]

    return n_agents


def setup_torch_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTorchPolicy  (w/o ValueNetworkMixin).
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def _algos_var(config):
    def _inner(key):
        expr = config['algo_args'][key]

        if not isinstance(expr, str):
            return expr
        else:
            float_express = (r'\d*\.\d*', float)
            sci_float = (r'\d+\.?\d*e-\d+|\d+\.\d*e\d+', float)
            sci_int = (r'\d+e\d+', lambda n: int(float(n)))
            bool_express = (r'True|False', lambda s: s == 'True')
            int_express = (r'\d+', int)

            express_matches = [
                float_express, sci_float, sci_int, bool_express, int_express
            ]

            value = expr

            for pat, type_f in express_matches:
                if re.search(pat, expr):
                    value = type_f(expr)
                    break

            return value

    return _inner


if __name__ == '__main__':
    _param = _algos_var({'algo_args': {'test': 'True'}})('test')

    assert _algos_var({'algo_args': {'test': False}})('test') is False
    assert _algos_var({'algo_args': {'test': 1}})('test') == 1
    assert _algos_var({'algo_args': {'test': 0.1}})('test') == 0.1
    assert _algos_var({'algo_args': {'test': '1e5'}})('test') == 100000
    assert _algos_var({'algo_args': {'test': '1e-5'}})('test') == 0.00001
    assert _algos_var({'algo_args': {'test': '1e0'}})('test') == 1
    assert _algos_var({'algo_args': {'test': '2e0'}})('test') == 2
    assert _algos_var({'algo_args': {'test': '1.01'}})('test') == 1.01
    assert _algos_var({'algo_args': {'test': '123'}})('test') == 123

    print('test done!')





