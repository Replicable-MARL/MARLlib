from ray.rllib.agents.qmix.qmix_policy import *
from ray.rllib.execution.replay_buffer import *
from ray.rllib.policy.sample_batch import SampleBatch

class QMixFromTorchPolicy(QMixTorchPolicy):

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config)

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
