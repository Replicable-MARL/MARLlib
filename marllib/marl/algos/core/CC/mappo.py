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

from ray.rllib.utils.framework import try_import_torch
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, ValueNetworkMixin, KLCoeffMixin, ppo_surrogate_loss
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from marllib.marl.algos.utils.centralized_critic import CentralizedValueMixin, centralized_critic_postprocessing

torch, nn = try_import_torch()



#############
### MAPPO ###
#############

# Copied from PPO but optimizing the central value function.
def setup_torch_mixins(policy, obs_space, action_space, config):
    # Copied from PPOTorchPolicy  (w/o ValueNetworkMixin).
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def central_critic_ppo_loss(policy, model, dist_class, train_batch):
    CentralizedValueMixin.__init__(policy)
    func = ppo_surrogate_loss

    vf_saved = model.value_function
    opp_action_in_cc = policy.config["model"]["custom_model_config"]["opp_action_in_cc"]
    model.value_function = lambda: policy.model.central_value_function(train_batch["state"],
                                                                       train_batch[
                                                                           "opponent_actions"] if opp_action_in_cc else None)

    policy._central_value_out = model.value_function()
    loss = func(policy, model, dist_class, train_batch)

    model.value_function = vf_saved

    return loss


MAPPOTorchPolicy = PPOTorchPolicy.with_updates(
    name="MAPPOTorchPolicy",
    get_default_config=lambda: PPO_CONFIG,
    postprocess_fn=centralized_critic_postprocessing,
    loss_fn=central_critic_ppo_loss,
    before_init=setup_torch_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        CentralizedValueMixin
    ])


def get_policy_class_mappo(config_):
    if config_["framework"] == "torch":
        return MAPPOTorchPolicy


MAPPOTrainer = PPOTrainer.with_updates(
    name="MAPPOTrainer",
    default_policy=None,
    get_policy_class=get_policy_class_mappo,
)

