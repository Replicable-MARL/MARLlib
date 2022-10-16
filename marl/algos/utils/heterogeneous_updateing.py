from ray.rllib.utils.framework import try_import_tf, try_import_torch, get_variable
import random
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_ops import explained_variance, sequence_mask
import re
from marl.algos.utils.centralized_critic_hetero import get_global_name, global_state_name
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
    #TODO : add document to explain the HA-train batch getitem mechanism. And make sure the document is fit
    """
    def __init__(self, main_train_batch, policy_name):
        self.main_train_batch = main_train_batch
        self.policy_name = policy_name

        self.copy = self.main_train_batch.copy
        self.keys = self.main_train_batch.keys
        self.is_training = self.main_train_batch.is_training

        self.pat = re.compile(r'^state_in_(\d+)')

    def get_state_index(self, string):
        match = self.pat.findall(string)
        if match:
            return match[0]
        else:
            return None

    def __getitem__(self, item):
        """
        Adds an adaptor to get the item.
        Input a key name, it would get the corresponding opponent's key-value
        """
        directly_get = [SampleBatch.SEQ_LENS]

        if item in directly_get:
            return self.main_train_batch[item]
        elif get_global_name(item, self.policy_name) in self.main_train_batch:
            return self.main_train_batch[get_global_name(item, self.policy_name)]
        elif state_index := self.get_state_index(item):
            return self.main_train_batch[global_state_name(state_index, self.policy_name)]

    def __contains__(self, item):
        if item in self.keys() or get_global_name(item, self.policy_name) in self.keys():
            return True
        elif state_index := self.get_state_index(item):
            if global_state_name(state_index, self.policy_name) in self.keys():
                return True

        return False


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
