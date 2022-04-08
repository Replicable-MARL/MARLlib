from ray.rllib.agents.qmix.qmix_policy import *
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.models.modelv2 import _unpack_obs
from ray.rllib.agents.qmix.model import RNNModel, _get_size
from gym.spaces import Tuple as Gym_Tuple, Discrete, Dict as Gym_Dict
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, List, Optional, TYPE_CHECKING
from ray.rllib.execution.replay_buffer import *

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.agents.qmix.qmix_policy import _mac, _validate
from SMAC.model.torch_vdn_qmix_iql_model import GRUModel, UPDeTModel


class QMixTorchPolicy_Customized(Policy):

    def __init__(self, obs_space, action_space, config):
        _validate(obs_space, action_space)
        config = dict(ray.rllib.agents.qmix.qmix.DEFAULT_CONFIG, **config)
        self.framework = "torch"
        super().__init__(obs_space, action_space, config)
        self.n_agents = len(obs_space.original_space.spaces)
        config["model"]["n_agents"] = self.n_agents
        self.n_actions = action_space.spaces[0].n
        self.h_size = config["model"]["lstm_cell_size"]
        self.has_env_global_state = False
        self.has_action_mask = False
        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))
        self.reward_standardize = config["reward_standardize"]

        agent_obs_space = obs_space.original_space.spaces[0]
        if isinstance(agent_obs_space, Gym_Dict):
            space_keys = set(agent_obs_space.spaces.keys())
            if "obs" not in space_keys:
                raise ValueError(
                    "Dict obs space must have subspace labeled `obs`")
            self.obs_size = _get_size(agent_obs_space.spaces["obs"])
            if "action_mask" in space_keys:
                mask_shape = tuple(agent_obs_space.spaces["action_mask"].shape)
                if mask_shape != (self.n_actions, ):
                    raise ValueError(
                        "Action mask shape must be {}, got {}".format(
                            (self.n_actions, ), mask_shape))
                self.has_action_mask = True
            if ENV_STATE in space_keys:
                self.env_global_state_shape = _get_size(
                    agent_obs_space.spaces[ENV_STATE])
                self.has_env_global_state = True
            else:
                self.env_global_state_shape = (self.obs_size, self.n_agents)
            # The real agent obs space is nested inside the dict
            config["model"]["full_obs_space"] = agent_obs_space
            agent_obs_space = agent_obs_space.spaces["obs"]
        else:
            self.obs_size = _get_size(agent_obs_space)
            self.env_global_state_shape = (self.obs_size, self.n_agents)

        neural_arch = config["model"]["custom_model_config"]["neural_arch"]
        self.model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space.spaces[0],
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=GRUModel if neural_arch == "GRU" else UPDeTModel).to(self.device)

        self.target_model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space.spaces[0],
            self.n_actions,
            config["model"],
            framework="torch",
            name="target_model",
            default_model=GRUModel if neural_arch == "GRU" else UPDeTModel).to(self.device)

        self.exploration = self._create_exploration()

        # Setup the mixer network.
        if config["mixer"] is None:  # "iql"
            self.mixer = None
            self.target_mixer = None
        elif config["mixer"] == "qmix":
            self.mixer = QMixer(self.n_agents, self.env_global_state_shape,
                                config["mixing_embed_dim"]).to(self.device)
            self.target_mixer = QMixer(
                self.n_agents, self.env_global_state_shape,
                config["mixing_embed_dim"]).to(self.device)
        elif config["mixer"] == "vdn":
            self.mixer = VDNMixer().to(self.device)
            self.target_mixer = VDNMixer().to(self.device)
        else:
            raise ValueError("Unknown mixer type {}".format(config["mixer"]))

        self.cur_epsilon = 1.0
        self.update_target()  # initial sync

        # Setup optimizer
        self.params = list(self.model.parameters())
        if self.mixer:
            self.params += list(self.mixer.parameters())
        self.loss = QMixLoss(self.model, self.target_model, self.mixer,
                             self.target_mixer, self.n_agents, self.n_actions,
                             self.config["double_q"], self.config["gamma"])

        if config["optimizer"] == "pymarl":
            from torch.optim import RMSprop
            self.optimiser = RMSprop(
                params=self.params,
                lr=config["lr"])

        elif config["optimizer"] == "epymarl":
            from torch.optim import Adam
            self.optimiser = Adam(
                params=self.params,
                lr=config["lr"],)

        else:
            raise ValueError("choose one optimizer type from pymarl(RMSprop) or epymarl(Adam)")

    @override(Policy)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        explore=None,
                        timestep=None,
                        **kwargs):
        explore = explore if explore is not None else self.config["explore"]
        obs_batch, action_mask, _ = self._unpack_observation(obs_batch)
        # We need to ensure we do not use the env global state
        # to compute actions

        # Compute actions
        with torch.no_grad():
            q_values, hiddens = _mac(
                self.model,
                torch.as_tensor(
                    obs_batch, dtype=torch.float, device=self.device), [
                    torch.as_tensor(
                        np.array(s), dtype=torch.float, device=self.device)
                    for s in state_batches
                ])
            avail = torch.as_tensor(
                action_mask, dtype=torch.float, device=self.device)
            masked_q_values = q_values.clone()
            masked_q_values[avail == 0.0] = -float("inf")
            masked_q_values_folded = torch.reshape(
                masked_q_values, [-1] + list(masked_q_values.shape)[2:])
            actions, _ = self.exploration.get_exploration_action(
                action_distribution=TorchCategorical(masked_q_values_folded),
                timestep=timestep,
                explore=explore)
            actions = torch.reshape(
                actions,
                list(masked_q_values.shape)[:-1]).cpu().numpy()
            hiddens = [s.cpu().numpy() for s in hiddens]

        return tuple(actions.transpose([1, 0])), hiddens, {}

    @override(Policy)
    def compute_log_likelihoods(self,
                                actions,
                                obs_batch,
                                state_batches=None,
                                prev_action_batch=None,
                                prev_reward_batch=None):
        obs_batch, action_mask, _ = self._unpack_observation(obs_batch)
        return np.zeros(obs_batch.size()[0])

    @override(Policy)
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

        # reduce the scale of reward for small variance. This is also
        # because we copy the global reward to each agent in rllib_env
        rewards = to_batches(rew, torch.float) / self.n_agents
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

    @override(Policy)
    def get_initial_state(self):  # initial RNN state
        return [
            s.expand([self.n_agents, -1]).cpu().numpy()
            for s in self.model.get_initial_state()
        ]

    @override(Policy)
    def get_weights(self):
        return {
            "model": self._cpu_dict(self.model.state_dict()),
            "target_model": self._cpu_dict(self.target_model.state_dict()),
            "mixer": self._cpu_dict(self.mixer.state_dict())
            if self.mixer else None,
            "target_mixer": self._cpu_dict(self.target_mixer.state_dict())
            if self.mixer else None,
        }

    @override(Policy)
    def set_weights(self, weights):
        self.model.load_state_dict(self._device_dict(weights["model"]))
        self.target_model.load_state_dict(
            self._device_dict(weights["target_model"]))
        if weights["mixer"] is not None:
            self.mixer.load_state_dict(self._device_dict(weights["mixer"]))
            self.target_mixer.load_state_dict(
                self._device_dict(weights["target_mixer"]))

    @override(Policy)
    def get_state(self):
        state = self.get_weights()
        state["cur_epsilon"] = self.cur_epsilon
        return state

    @override(Policy)
    def set_state(self, state):
        self.set_weights(state)
        self.set_epsilon(state["cur_epsilon"])

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        logger.debug("Updated target networks")
        print("Updated target networks")

    def set_epsilon(self, epsilon):
        self.cur_epsilon = epsilon

    def _get_group_rewards(self, info_batch):
        group_rewards = np.array([
            info.get(GROUP_REWARDS, [0.0] * self.n_agents)
            for info in info_batch
        ])
        return group_rewards

    def _device_dict(self, state_dict):
        return {
            k: torch.as_tensor(v, device=self.device)
            for k, v in state_dict.items()
        }

    @staticmethod
    def _cpu_dict(state_dict):
        return {k: v.cpu().detach().numpy() for k, v in state_dict.items()}

    def _unpack_observation(self, obs_batch):
        """Unpacks the observation, action mask, and state (if present)
        from agent grouping.

        Returns:
            obs (np.ndarray): obs tensor of shape [B, n_agents, obs_size]
            mask (np.ndarray): action mask, if any
            state (np.ndarray or None): state tensor of shape [B, state_size]
                or None if it is not in the batch
        """

        unpacked = _unpack_obs(
            np.array(obs_batch, dtype=np.float32),
            self.observation_space.original_space,
            tensorlib=np)

        if isinstance(unpacked[0], dict):
            assert "obs" in unpacked[0]
            unpacked_obs = [
                np.concatenate(tree.flatten(u["obs"]), 1) for u in unpacked
            ]
        else:
            unpacked_obs = unpacked

        obs = np.concatenate(
            unpacked_obs,
            axis=1).reshape([len(obs_batch), self.n_agents, self.obs_size])

        if self.has_action_mask:
            action_mask = np.concatenate(
                [o["action_mask"] for o in unpacked], axis=1).reshape(
                [len(obs_batch), self.n_agents, self.n_actions])
        else:
            action_mask = np.ones(
                [len(obs_batch), self.n_agents, self.n_actions],
                dtype=np.float32)

        if self.has_env_global_state:
            state = np.concatenate(tree.flatten(unpacked[0][ENV_STATE]), 1)
        else:
            state = None
        return obs, action_mask, state


# customized the LocalReplayBuffer to ensure the return batchsize = 32
# be aware, although the rllib doc says, capacity is sequence number not one step data when replay_sequence_length > 1,
# in fact, it is not when sampling batch. This might be a bug for
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