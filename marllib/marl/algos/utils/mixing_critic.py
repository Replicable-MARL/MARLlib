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

from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from copy import deepcopy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.numpy import convert_to_numpy
import numpy as np
from marllib.marl.algos.utils.centralized_Q import get_dim
from marllib.marl.algos.utils.mixing_Q import align_batch

torch, nn = try_import_torch()

"""
mixing critic postprocessing for 
1. VDA2C 
2. VDPPO 
"""


class MixingValueMixin:

    def __init__(self):
        self.mixing_vf = "mixing"


# get opponent value vf
def value_mixing_postprocessing(policy,
                                sample_batch,
                                other_agent_batches=None,
                                episode=None):
    custom_config = policy.config["model"]["custom_model_config"]
    algorithm = custom_config["algorithm"]
    pytorch = custom_config["framework"] == "torch"
    obs_dim = get_dim(custom_config["space_obs"]["obs"].shape)  # 3d input this is channel dim of obs
    if custom_config["global_state_flag"]:
        state_dim = get_dim(custom_config["space_obs"]["state"].shape)
    else:
        state_dim = None

    if custom_config["mask_flag"]:
        action_mask_dim = custom_config["space_act"].n
    else:
        action_mask_dim = 0
    n_agents = custom_config["num_agents"]
    opponent_agents_num = n_agents - 1

    if (pytorch and hasattr(policy, "mixing_vf")) or \
            (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None
        opponent_batch_list = list(other_agent_batches.values())
        raw_opponent_batch = [opponent_batch_list[i][1] for i in range(opponent_agents_num)]
        opponent_batch = []
        for one_opponent_batch in raw_opponent_batch:
            one_opponent_batch = align_batch(one_opponent_batch, sample_batch)
            opponent_batch.append(one_opponent_batch)

        if state_dim:
            sample_batch["state"] = sample_batch['obs'][:, action_mask_dim + obs_dim:]
        else:  # all other agent obs as state
            # sample_batch["state"] = sample_batch['obs'][:, action_mask_dim:action_mask_dim + obs_dim]
            sample_batch["state"] = np.stack(
                [sample_batch['obs'][:, action_mask_dim:action_mask_dim + obs_dim]] + [
                    opponent_batch[i]["obs"][:, action_mask_dim:action_mask_dim + obs_dim] for i in
                    range(opponent_agents_num)], 1)

        sample_batch["opponent_vf_preds"] = np.stack(
            [opponent_batch[i]["vf_preds"] for i in range(opponent_agents_num)], 1)

        sample_batch["all_vf_preds"] = np.concatenate(
            (np.expand_dims(sample_batch["vf_preds"], axis=1), sample_batch["opponent_vf_preds"]), axis=1)

        sample_batch["vf_tot"] = convert_to_numpy(policy.model.mixing_value(
            convert_to_torch_tensor(sample_batch["all_vf_preds"], policy.device),
            convert_to_torch_tensor(sample_batch["state"], policy.device)))

    else:
        # Policy hasn't been initialized yet, use zeros.
        o = sample_batch[SampleBatch.CUR_OBS]
        if state_dim:
            sample_batch["state"] = np.zeros((o.shape[0], state_dim),
                                             dtype=sample_batch[SampleBatch.CUR_OBS].dtype)
        else:
            sample_batch["state"] = np.zeros((o.shape[0], n_agents, obs_dim),
                                             dtype=sample_batch[SampleBatch.CUR_OBS].dtype)

        sample_batch["vf_preds"] = np.zeros_like(
            sample_batch[SampleBatch.REWARDS], dtype=np.float32)
        sample_batch["opponent_vf_preds"] = np.zeros(
            (sample_batch["vf_preds"].shape[0], opponent_agents_num),
            dtype=sample_batch["obs"].dtype)
        sample_batch["all_vf_preds"] = np.concatenate(
            (np.expand_dims(sample_batch["vf_preds"], axis=1), sample_batch["opponent_vf_preds"]), axis=1)

        sample_batch["vf_tot"] = convert_to_numpy(policy.model.mixing_value(
            convert_to_torch_tensor(sample_batch["all_vf_preds"], policy.device),
            convert_to_torch_tensor(sample_batch["state"], policy.device)))

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        # if isinstance(sample_batch["vf_tot"], float):
        #     print(1)
        last_r = sample_batch["vf_tot"][-1]

    train_batch = compute_advantages_vf_tot(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return train_batch


def compute_advantages_vf_tot(rollout: SampleBatch,
                              last_r: float,
                              gamma: float = 0.9,
                              lambda_: float = 1.0,
                              use_gae: bool = True,
                              use_critic: bool = True):
    # save the original vf
    vf_saved = deepcopy(rollout[SampleBatch.VF_PREDS])
    rollout[SampleBatch.VF_PREDS] = rollout["vf_tot"]
    rollout = compute_advantages(
        rollout,
        last_r,
        gamma,
        lambda_,
        use_gae,
        use_critic)
    rollout[SampleBatch.VF_PREDS] = vf_saved

    return rollout
