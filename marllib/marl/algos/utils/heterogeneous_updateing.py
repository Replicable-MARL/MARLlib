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
import random
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_ops import sequence_mask
import re
from marllib.marl.algos.utils.centralized_critic_hetero import get_global_name, global_state_name

torch, nn = try_import_torch()


def get_mask_and_reduce_mean(model, train_batch, dist_class):
    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)

    # RNN case: Mask away 0-padded chunks at end of time axis.
    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major())
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean

    return mask, reduce_mean_valid, curr_action_dist


def update_m_advantage(iter_model, iter_train_batch, iter_dist_class, iter_prev_action_logp, iter_actions, m_advantage):
    with torch.no_grad():
        iter_model.eval()
        iter_new_logits, _ = iter_model(iter_train_batch)
        try:
            iter_new_action_dist = iter_dist_class(iter_new_logits, iter_model)
            iter_new_logp_ratio = torch.exp(
                iter_new_action_dist.logp(iter_actions) -
                iter_prev_action_logp
            )
        except ValueError as e:
            print(e)

    m_advantage = iter_new_logp_ratio * m_advantage

    return m_advantage


class IterTrainBatch(SampleBatch):
    """
    This is an adaptor for heterogeneous updating.
    At firstly, We add the opponent or collaborator information, which includes [actions, obs, states, .. etc], into
    train_batch in post-processing method.

    When update individual model in heterogeneous, we need the following evaluation:

        logits, states = model(train_batch), the train_batch is the train information of one specific agent.

    therefore, the complete statement is the following:

        ith_logits, ith_states = ith_model(ith_train_batch)

    It means, if we have n agents, we want to update ith agent, we need give the ith agent's train batch to ith model.
    we can get the ith model by

    Reconstruction a new individual agent-wise train batch from the post processed is difficult, actually.

    We could build an adaptor to solve this.

    the ith_train_batch will be created by IterTrainBatch(train_batch, policy_name), named as iter_batch in the following.

    after created, we redefine the __getitem__ and __contains__ methods.

    in __getitem__, if  iter_batch get one key, such as iter_batch['action'], the really thing will occur is that the
    train_batch['<policy_name>_action'] will be gotten. And also like other keys.

    the __contains__ also performs like above, if you want to test 'action' in iter_batch, will not test the 'action', but
    test the '<policy_name>_action' in train_batch.

    The benefits about this Adaptor is that we will not modify the actor model.

    If not by this way, we need to refactor the calculation process in actor forward() or need to reconstruction an iter
    train batch.
    """
    def __init__(self, main_train_batch, policy_name):
        self.main_train_batch = main_train_batch
        self.policy_name = policy_name

        self.copy = self.main_train_batch.copy
        self.keys = self.main_train_batch.keys
        self.is_training = self.main_train_batch.is_training

        self.pat = re.compile(r'^state_in_(\d+)')

def get_each_agent_train(model, policy, dist_class, train_batch):
    all_policies_with_names = list(model.other_policies.items()) + [('self', policy)]
    random.shuffle(all_policies_with_names)

    for policy_name, iter_policy in all_policies_with_names:
        is_self = (policy_name == 'self')
        iter_model = [iter_policy.model, model][is_self]
        iter_dist_class = [iter_policy.dist_class, dist_class][is_self]
        iter_train_batch = [IterTrainBatch(train_batch, policy_name), train_batch][is_self]
        iter_mask, iter_reduce_mean, current_action_dist = get_mask_and_reduce_mean(iter_model, iter_train_batch, dist_class)
        iter_actions = iter_train_batch[SampleBatch.ACTIONS]
        iter_prev_action_logp = iter_train_batch[SampleBatch.ACTION_LOGP]

        yield iter_model, iter_dist_class, iter_train_batch, iter_mask, iter_reduce_mean, iter_actions, iter_policy, iter_prev_action_logp
