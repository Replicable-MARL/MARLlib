import numpy as np
from typing import Dict, List, Any, Union
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch, \
    TensorType
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from GRF.model.torch_cnn_updet import Transformer

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class Torch_CNN_Transformer_CentralizedCritic_Model(TorchRNN, nn.Module):

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            emb=32,
            heads=3,
            depth=2,
            **kwargs,
    ):
        self.conv_emb = 4 * 7 * 7
        self.trans_emb = emb
        self.heads = heads
        self.depth = depth
        self.agent_num = model_config["custom_model_config"]["agent_num"]
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # Build the Module from Conv + FC + GRU + 2xfc (action + value outs).
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=2,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=3),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=4,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.fc1 = nn.Linear(self.conv_emb, self.trans_emb)  # fully connected layer, output 10 classes

        self._features = None

        # Build the Module from Transformer / regard as RNN
        self.transformer = Transformer(self.trans_emb, self.heads, self.depth, self.trans_emb)
        self.action_branch = nn.Linear(self.trans_emb, num_outputs)
        self.value_branch = nn.Linear(self.trans_emb, 1)

        # cc value for additional teammate actions
        self.value_branch_cc = nn.Linear(self.trans_emb * 4 + self.agent_num - 1, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.action_branch.weight.new(1, self.trans_emb).zero_().squeeze(0),
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
        lstm_length = inputs.shape[1]  # mimic lstm logic in rllib
        seq_output = []
        feature_output = []
        for i in range(lstm_length):
            output, state = self.forward_rnn(inputs[:, i:i + 1, :], state, seq_lens)
            seq_output.append(output)
            # record self._feature
            feature_output.append(torch.flatten(self._features, start_dim=1))

        output = torch.stack(seq_output, dim=1)
        output = torch.reshape(output, [-1, self.num_outputs])

        # record self._feature for self.value_function()
        self._features = torch.stack(feature_output, dim=1)

        return output, state

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        b = inputs.shape[0]
        x = inputs.permute(0, 1, 4, 2, 3).reshape(-1, 1, 42, 42)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x.view(4 * b, -1))
        x = x.view(b, 4, self.trans_emb)

        # inputs = self._build_inputs_transformer(inputs)
        outputs, _ = self.transformer(x, state[0], None)

        # last dim for hidden state
        h = outputs[:, -1:, :]

        # record self._features
        self._features = torch.max(outputs[:, :-1, :], 1)[0]
        logits = self.action_branch(self._features)

        # Return masked logits.
        return logits, [torch.squeeze(h, 1)]

    def _build_inputs_transformer(self, inputs):
        pos = 4 - self.token_dim  # 5 for -1 6 for -2
        arranged_obs = torch.cat((inputs[:, :, pos:], inputs[:, :, :pos]), 2)
        reshaped_obs = arranged_obs.view(-1, 1 + (self.enemy_num - 1) + self.ally_num, self.token_dim)

        return reshaped_obs

    def central_value_function(self, obs, opponent_actions):
        obs = obs.float()
        b = obs.shape[0]
        x = obs.permute(0, 3, 1, 2).reshape(-1, 1, 42, 42)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x.view(4 * b, -1))
        x = x.view(b, 4 * self.trans_emb)
        x = torch.cat((x, opponent_actions), 1)
        return torch.reshape(self.value_branch_cc(x), [-1])


