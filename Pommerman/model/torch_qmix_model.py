from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.agents.qmix.qmix_policy import QMixTorchPolicy, QMixLoss
from gym.spaces import Tuple, Discrete, Dict
from ray.rllib.agents.qmix.mixers import VDNMixer, QMixer
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.agents.qmix.qmix import *
# from GoogleFootball.model.torch_cnn_updet import Transformer

torch, nn = try_import_torch()


class Torch_CNN_GRU_Model(TorchModelV2, nn.Module):
    """The GRU model only for QMIX."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, hidden_state_size=256,):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.hidden_state_size = hidden_state_size
        self.n_agents = model_config["n_agents"]
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
        self.rnn = nn.GRUCell(self.hidden_state_size, self.hidden_state_size)
        self.fc3 = nn.Linear(self.hidden_state_size, num_outputs)

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        return [
            self.fc1.weight.new(self.n_agents,
                                self.hidden_state_size).zero_().squeeze(0)
        ]

    @override(ModelV2)
    def forward(self, input_dict, hidden_state, seq_lens):
        obs = input_dict["obs"][:, :-4].view(-1, self.map_size, self.map_size, 5).float()
        status = input_dict["obs"][:, -4:].float()

        x = self.conv1(obs.view(-1, obs.shape[1], obs.shape[2], obs.shape[3]).permute(0, 3, 1, 2))
        x = self.conv2(x)
        x_obs = nn.functional.relu(self.fc1(x.view(obs.shape[0], -1)))
        x_status = nn.functional.relu(self.fc2(status))
        x = torch.cat((x_obs, x_status), 1)

        h_in = hidden_state[0].reshape(-1, self.hidden_state_size)
        h = self.rnn(x, h_in)
        q = self.fc3(h)
        return q, [h]


def _get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size

