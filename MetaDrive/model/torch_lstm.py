
import numpy as np
from typing import Dict, List, Any, Union
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch, \
    TensorType
from ray.rllib.policy.rnn_sequencing import add_time_dimension


tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class Torch_LSTM_Model(TorchRNN, nn.Module):

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            fc_size=64,
            lstm_state_size=256,
            **kwargs,
    ):
        self.obs_size = obs_space.shape[0]
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # Build the Module from fc + LSTM + 2xfc (action + value outs).
        self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        self.lstm = nn.LSTM(
            self.fc_size, self.lstm_state_size, batch_first=True)
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        h = [
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
            self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0)
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(ModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        flat_inputs = input_dict["obs"].float()
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        max_seq_len = flat_inputs.shape[0] // seq_lens.shape[0]
        self.time_major = self.model_config.get("_time_major", False)
        inputs = add_time_dimension(
            flat_inputs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )
        output, new_state = self.forward_rnn(inputs, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])

        return output, new_state

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        # Extract the available actions tensor from the observation.

        # Compute the unmasked logits.
        x = nn.functional.relu(self.fc1(inputs))
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0),
                torch.unsqueeze(state[1], 0)])
        logits = self.action_branch(self._features)

        # Return masked logits.
        return logits, [torch.squeeze(h, 0), torch.squeeze(c, 0)]