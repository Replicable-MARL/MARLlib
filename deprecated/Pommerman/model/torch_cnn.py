import numpy as np
from typing import Dict, List, Any, Union
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch, \
    TensorType
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class Torch_CNN_Model(TorchModelV2, nn.Module):
    """The GRU model only for QMIX."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, hidden_state_size=256,):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.hidden_state_size = hidden_state_size
        self.n_agents = model_config["custom_model_config"]["agent_num"]
        self.map_size = model_config["custom_model_config"]["map_size"]

        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=5,
                out_channels=8,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.fc1 = nn.Linear(64, self.hidden_state_size // 2)
        self.fc2 = nn.Linear(4, self.hidden_state_size // 2)  # fully connected layer, output 10 classes
        self.action_branch = nn.Linear(self.hidden_state_size, num_outputs)
        self.value_branch = nn.Linear(self.hidden_state_size, 1)

    @override(ModelV2)
    def forward(self, input_dict, hidden_state, seq_lens):
        obs = input_dict["obs"]["obs"].float()
        status = input_dict["obs"]["status"].float()

        x = self.conv1(obs.permute(0, 3, 1, 2))
        x = self.conv2(x)
        x_obs = nn.functional.relu(self.fc1(x.view(obs.shape[0], -1)))
        x_status = nn.functional.relu(self.fc2(status))
        self._features = torch.cat((x_obs, x_status), 1)

        q = self.action_branch(self._features)
        return q, []

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])
