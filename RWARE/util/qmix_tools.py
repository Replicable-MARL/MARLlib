from ray.rllib.agents.qmix.qmix_policy import *
from ray.rllib.execution.replay_buffer import *
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.agents.qmix.qmix_policy import _mac, _validate, _unroll_mac


# the _unroll_mac for next observation is different from Pymarl.
# we provide a new QMixloss here
class Pymarl_QMixLoss(nn.Module):
    def __init__(self,
                 model,
                 target_model,
                 mixer,
                 target_mixer,
                 n_agents,
                 n_actions,
                 double_q=True,
                 gamma=0.99):
        nn.Module.__init__(self)
        self.model = model
        self.target_model = target_model
        self.mixer = mixer
        self.target_mixer = target_mixer
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.double_q = double_q
        self.gamma = gamma

    def forward(self,
                rewards,
                actions,
                terminated,
                mask,
                obs,
                next_obs,
                action_mask,
                next_action_mask,
                state=None,
                next_state=None):
        """Forward pass of the loss.

        Args:
            rewards: Tensor of shape [B, T, n_agents]
            actions: Tensor of shape [B, T, n_agents]
            terminated: Tensor of shape [B, T, n_agents]
            mask: Tensor of shape [B, T, n_agents]
            obs: Tensor of shape [B, T, n_agents, obs_size]
            next_obs: Tensor of shape [B, T, n_agents, obs_size]
            action_mask: Tensor of shape [B, T, n_agents, n_actions]
            next_action_mask: Tensor of shape [B, T, n_agents, n_actions]
            state: Tensor of shape [B, T, state_dim] (optional)
            next_state: Tensor of shape [B, T, state_dim] (optional)
        """

        # Assert either none or both of state and next_state are given
        if state is None and next_state is None:
            state = obs  # default to state being all agents' observations
            next_state = next_obs
        elif (state is None) != (next_state is None):
            raise ValueError("Expected either neither or both of `state` and "
                             "`next_state` to be given. Got: "
                             "\n`state` = {}\n`next_state` = {}".format(
                state, next_state))

        # append the first element of obs + next_obs to get new one
        whole_obs = torch.cat((obs[:, 0:1], next_obs), axis=1)

        # Calculate estimated Q-Values
        mac_out = _unroll_mac(self.model, whole_obs)

        # Pick the Q-Values for the actions taken -> [B * n_agents, T]
        chosen_action_qvals = torch.gather(
            mac_out[:, :-1], dim=3, index=actions.unsqueeze(3)).squeeze(3)

        # Calculate the Q-Values necessary for the target

        target_mac_out = _unroll_mac(self.target_model, whole_obs)

        # we only need target_mac_out for raw next_obs part
        target_mac_out = target_mac_out[:, 1:]

        # Mask out unavailable actions for the t+1 step
        ignore_action_tp1 = (next_action_mask == 0) & (mask == 1).unsqueeze(-1)
        target_mac_out[ignore_action_tp1] = -np.inf

        # Max over target Q-Values
        if self.double_q:
            # large bugs here in original QMixloss, the gradient is calculated
            # we fix this follow pymarl
            mac_out_tp1 = mac_out.clone().detach()
            mac_out_tp1 = mac_out_tp1[:, 1:]

            # mask out unallowed actions
            mac_out_tp1[ignore_action_tp1] = -np.inf

            # obtain best actions at t+1 according to policy NN
            cur_max_actions = mac_out_tp1.argmax(dim=3, keepdim=True)

            # use the target network to estimate the Q-values of policy
            # network's selected actions
            target_max_qvals = torch.gather(target_mac_out, 3,
                                            cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        assert target_max_qvals.min().item() != -np.inf, \
            "target_max_qvals contains a masked action; \
            there may be a state with no valid actions."

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, state)
            target_max_qvals = self.target_mixer(target_max_qvals, next_state)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        return loss, mask, masked_td_error, chosen_action_qvals, targets

class QMixTorchPolicy_Customized(QMixTorchPolicy):

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config)
        self.reward_standardize = config["reward_standardize"]

        self.loss = Pymarl_QMixLoss(self.model, self.target_model, self.mixer,
                                    self.target_mixer, self.n_agents, self.n_actions,
                                    self.config["double_q"], self.config["gamma"])

        if config["optimizer"] == "RMSprop":
            from torch.optim import RMSprop
            self.optimiser = RMSprop(
                params=self.params,
                lr=config["lr"],
                alpha=config["optim_alpha"],
                eps=config["optim_eps"])

        elif config["optimizer"] == "Adam":
            from torch.optim import Adam
            self.optimiser = Adam(
                params=self.params,
                lr=config["lr"],
                eps=config["optim_eps"])

        else:
            raise ValueError("choose one optimizer type from RMSprop or Adam")

    @override(QMixTorchPolicy)
    def learn_on_batch(self, samples):
        obs_batch, action_mask, env_global_state = self._unpack_observation(
            samples[SampleBatch.CUR_OBS])
        (next_obs_batch, next_action_mask,
         next_env_global_state) = self._unpack_observation(
            samples[SampleBatch.NEXT_OBS])
        group_rewards = self._get_group_rewards(samples[SampleBatch.INFOS])

        input_list = [
            group_rewards, action_mask, next_action_mask,
            samples[SampleBatch.ACTIONS], samples[SampleBatch.DONES],
            obs_batch, next_obs_batch
        ]
        if self.has_env_global_state:
            input_list.extend([env_global_state, next_env_global_state])

        output_list, _, seq_lens = \
            chop_into_sequences(
                episode_ids=samples[SampleBatch.EPS_ID],
                unroll_ids=samples[SampleBatch.UNROLL_ID],
                agent_indices=samples[SampleBatch.AGENT_INDEX],
                feature_columns=input_list,
                state_columns=[],  # RNN states not used here
                max_seq_len=self.config["model"]["max_seq_len"],
                dynamic_max=True)
        # These will be padded to shape [B * T, ...]
        if self.has_env_global_state:
            (rew, action_mask, next_action_mask, act, dones, obs, next_obs,
             env_global_state, next_env_global_state) = output_list
        else:
            (rew, action_mask, next_action_mask, act, dones, obs,
             next_obs) = output_list
        B, T = len(seq_lens), max(seq_lens)

        def to_batches(arr, dtype):
            new_shape = [B, T] + list(arr.shape[1:])
            return torch.as_tensor(
                np.reshape(arr, new_shape), dtype=dtype, device=self.device)

        rewards = to_batches(rew, torch.float)
        if self.reward_standardize:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        actions = to_batches(act, torch.long)
        obs = to_batches(obs, torch.float).reshape(
            [B, T, self.n_agents, self.obs_size])
        action_mask = to_batches(action_mask, torch.float)
        next_obs = to_batches(next_obs, torch.float).reshape(
            [B, T, self.n_agents, self.obs_size])
        next_action_mask = to_batches(next_action_mask, torch.float)
        if self.has_env_global_state:
            env_global_state = to_batches(env_global_state, torch.float)
            next_env_global_state = to_batches(next_env_global_state,
                                               torch.float)

        # TODO(ekl) this treats group termination as individual termination
        terminated = to_batches(dones, torch.float).unsqueeze(2).expand(
            B, T, self.n_agents)

        # Create mask for where index is < unpadded sequence length
        filled = np.reshape(
            np.tile(np.arange(T, dtype=np.float32), B),
            [B, T]) < np.expand_dims(seq_lens, 1)
        mask = torch.as_tensor(
            filled, dtype=torch.float, device=self.device).unsqueeze(2).expand(
            B, T, self.n_agents)

        # Compute loss
        loss_out, mask, masked_td_error, chosen_action_qvals, targets = (
            self.loss(rewards, actions, terminated, mask, obs, next_obs,
                      action_mask, next_action_mask, env_global_state,
                      next_env_global_state))

        # Optimise
        self.optimiser.zero_grad()
        loss_out.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.params, self.config["grad_norm_clipping"])
        self.optimiser.step()

        mask_elems = mask.sum().item()
        stats = {
            "loss": loss_out.item(),
            "grad_norm": grad_norm
            if isinstance(grad_norm, float) else grad_norm.item(),
            "td_error_abs": masked_td_error.abs().sum().item() / mask_elems,
            "q_taken_mean": (chosen_action_qvals * mask).sum().item() /
                            mask_elems,
            "target_mean": (targets * mask).sum().item() / mask_elems,
        }
        return {LEARNER_STATS_KEY: stats}


# customized the LocalReplayBuffer to ensure the return batchsize = 32
# be aware, although the rllib doc says, capacity here is the sequence number when replay_sequence_length > 1,
# in fact, it is not. This might be a bug and my have been fixed in the future version
class QMixReplayBuffer(LocalReplayBuffer):

    def __init__(
            self,
            num_shards: int = 1,
            learning_starts: int = 1000,
            capacity: int = 10000,
            replay_batch_size: int = 32,
            prioritized_replay_alpha: float = 0.6,
            prioritized_replay_beta: float = 0.4,
            prioritized_replay_eps: float = 1e-6,
            replay_mode: str = "independent",
            replay_sequence_length: int = 1,
            replay_burn_in: int = 0,
            replay_zero_init_states: bool = True,
            buffer_size=DEPRECATED_VALUE,
    ):
        LocalReplayBuffer.__init__(self, num_shards, learning_starts, capacity, replay_batch_size,
                                   prioritized_replay_alpha, prioritized_replay_beta,
                                   prioritized_replay_eps, replay_mode, replay_sequence_length, replay_burn_in,
                                   replay_zero_init_states,
                                   buffer_size)

        self.replay_batch_size = replay_batch_size
        # self.len_debug = 0

    @override(LocalReplayBuffer)
    def add_batch(self, batch: SampleBatchType) -> None:
        # Make a copy so the replay buffer doesn't pin plasma memory.
        batch = batch.copy()
        # Handle everything as if multiagent
        if isinstance(batch, SampleBatch):
            batch = MultiAgentBatch({DEFAULT_POLICY_ID: batch}, batch.count)

        with self.add_batch_timer:
            # Lockstep mode: Store under _ALL_POLICIES key (we will always
            # only sample from all policies at the same time).
            for policy_id, sample_batch in batch.policy_batches.items():
                timeslices = [sample_batch]
                for time_slice in timeslices:
                    # If SampleBatch has prio-replay weights, average
                    # over these to use as a weight for the entire
                    # sequence.
                    if "weights" in time_slice and \
                            len(time_slice["weights"]):
                        weight = np.mean(time_slice["weights"])
                    else:
                        weight = None
                    self.replay_buffers[policy_id].add(
                        time_slice, weight=weight)
        self.num_added += batch.count
        # if self.len_debug != len(self.replay_buffers["default_policy"]):
        #     self.len_debug = len(self.replay_buffers["default_policy"])
        # else:
        #     print(1)
