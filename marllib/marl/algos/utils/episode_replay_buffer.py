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

from ray.rllib.execution.replay_buffer import *


class EpisodeBasedReplayBuffer(LocalReplayBuffer):

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
            for policy_id, sample_batch in batch.policy_batches.items():
                timeslices = [sample_batch]
                for time_slice in timeslices:
                    if "weights" in time_slice and \
                            len(time_slice["weights"]):
                        weight = np.mean(time_slice["weights"])
                    else:
                        weight = None
                    self.replay_buffers[policy_id].add(
                        time_slice, weight=weight)
        self.num_added += batch.count
