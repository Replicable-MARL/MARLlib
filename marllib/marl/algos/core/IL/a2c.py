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

from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.a3c.a2c import A2C_DEFAULT_CONFIG as A2C_CONFIG, A2CTrainer

###########
### A2C ###
###########


IA2CTorchPolicy = A3CTorchPolicy.with_updates(
    name="IA2CTorchPolicy",
    get_default_config=lambda: A2C_CONFIG,
)


def get_policy_class_ia2c(config_):
    if config_["framework"] == "torch":
        return IA2CTorchPolicy


IA2CTrainer = A2CTrainer.with_updates(
    name="IA2CTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_ia2c,
)
