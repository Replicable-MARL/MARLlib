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


class Torch_CNN_LSTM_CentralizedCritic_Model(TorchRNN, nn.Module):

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
        # self.obs_size = obs_space.shape[0]
        self.fc_size = fc_size
        self.lstm_state_size = lstm_state_size
        self.agent_num = model_config["custom_model_config"]["agent_num"]
        self.mini_channel_dim = model_config["custom_model_config"]["mini_channel_dim"]

        self.obs_scope = obs_space.shape[0]
        self.channel_dim = obs_space.shape[2] - self.mini_channel_dim
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # Build the Module from Conv + FC + GRU + 2xfc (action + value outs).
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=self.channel_dim,
                out_channels=8,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        out_dim = self.obs_scope // 4
        self.fc1 = nn.Linear(16 * out_dim * out_dim, self.fc_size)
        self.lstm = nn.LSTM(
            self.fc_size, self.lstm_state_size, batch_first=True)
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)

        # cc value for additional teammate actions
        self.conv1_cc = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=self.channel_dim + self.mini_channel_dim,
                out_channels=8,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
            ),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2_cc = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        out_dim = self.obs_scope // 4
        self.fc1_cc = nn.Linear(16 * out_dim * out_dim, self.fc_size)
        self.value_branch_cc = nn.Linear(self.fc_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

        # coma needs a central_vf with action number output
        self.coma_flag = False
        if "coma" in model_config["custom_model_config"]:
            self.coma_flag = True
            self.value_branch_cc = nn.Linear(self.fc_size, num_outputs)

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
        obs = inputs.view(-1, inputs.shape[2], inputs.shape[3], inputs.shape[4]).permute(0, 3, 1, 2)
        # refer to https://www.pettingzoo.ml/magent, eliminate the minimap from the obs inputs
        obs = obs[:, :self.channel_dim]
        x = self.conv1(obs)
        x = self.conv2(x)
        x = nn.functional.relu(self.fc1(x.view(inputs.shape[0], inputs.shape[1], -1)))
        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0),
                torch.unsqueeze(state[1], 0)])
        logits = self.action_branch(self._features)

        return logits, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    def central_value_function(self, inputs):

        x = self.conv1_cc(inputs.permute(0, 3, 1, 2))
        x = self.conv2_cc(x)
        x = nn.functional.relu(self.fc1_cc(x.view(inputs.shape[0], -1)))
        if self.coma_flag:
            return torch.reshape(self.value_branch_cc(x), [-1, self.num_outputs])
        else:
            return torch.reshape(self.value_branch_cc(x), [-1])
