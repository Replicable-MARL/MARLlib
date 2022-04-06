from ray.rllib.agents.qmix.qmix_policy import *
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.models.modelv2 import _unpack_obs
from ray.rllib.agents.qmix.model import RNNModel, _get_size
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from gym.spaces import Tuple as Gym_Tuple, Discrete, Dict as Gym_Dict

import numpy as np
import tree  # pip install dm_tree
from typing import Dict, List, Optional, TYPE_CHECKING

from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.utils.typing import AgentID, ModelGradients, ModelWeights, \
    TensorType, TensorStructType, TrainerConfigDict, Tuple, Union

from SMAC.model.torch_qmix_mask_gru_updet import GRUModel


class QMixFromTorchPolicy(QMixTorchPolicy):

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config)
        config["model"]["n_agents"] = self.n_agents

        agent_obs_space = obs_space.original_space.spaces[0]
        if isinstance(agent_obs_space, Gym_Dict):
            space_keys = set(agent_obs_space.spaces.keys())
            if "obs" not in space_keys:
                raise ValueError(
                    "Dict obs space must have subspace labeled `obs`")
            self.obs_size = _get_size(agent_obs_space.spaces["obs"])
            if "action_mask" in space_keys:
                mask_shape = tuple(agent_obs_space.spaces["action_mask"].shape)
                if mask_shape != (self.n_actions,):
                    raise ValueError(
                        "Action mask shape must be {}, got {}".format(
                            (self.n_actions,), mask_shape))
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
            default_model=GRUModel).to(self.device)

        self.target_model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space.spaces[0],
            self.n_actions,
            config["model"],
            framework="torch",
            name="target_model",
            default_model=GRUModel).to(self.device)

        self.exploration = self._create_exploration()

        # Setup the mixer network.
        if config["mixer"] is None: # "iql"
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
        from torch.optim import RMSprop
        # self.optimiser = RMSprop(
        #     params=self.params,
        #     lr=config["lr"],
        #     alpha=config["optim_alpha"],
        #     eps=config["optim_eps"])
        from torch.optim import Adam
        self.optimiser = Adam(
            params=self.params,
            lr=config["lr"],
            eps=config["optim_eps"])


from ray.rllib.execution.replay_buffer import *


# customized the LocalReplayBuffer to ensure the return batchsize = 32
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

