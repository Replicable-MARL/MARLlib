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

"""
examples on how to grid search based on MARLlib api:
    1. learning rate
    2. model layer dimension

Notes:
    1. local_mode must set to (False) to enable parallelized training;
    2. other tunable parameters can be found in model arch and algorithm config
    3. grid_search function not available on: trpo family and joint Q family
"""

from marllib import marl
from ray import tune

env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)

mappo = marl.algos.mappo(hyperparam_source="test", lr=tune.grid_search([0.0005, 0.001]))

model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": tune.grid_search(["8-16", "16-32"])})

mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 20000000}, local_mode=False, num_gpus=0,
          num_workers=1, share_policy='all', checkpoint_freq=500)

# more examples on ray search spaces can be found at this link:
# https://docs.ray.io/en/latest/tune/tutorials/tune-search-spaces.html
