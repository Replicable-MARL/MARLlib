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

import re
from ray.rllib.agents.ppo.ppo_torch_policy import KLCoeffMixin
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


def get_device():
    if torch.cuda.is_available():
        return f'cuda:{torch.cuda.current_device()}'
    else:
        return 'cpu'


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


class AlgVar(dict):
    def __init__(self, base_dict: dict, key="algo_args"):
        key = key or list(base_dict.keys())[0]
        super().__init__(base_dict[key])

    def __getitem__(self, item):
        expr = self.get(item, None)

        if expr is None: raise KeyError(f'{item} not exists')
        elif not isinstance(expr, str):
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





