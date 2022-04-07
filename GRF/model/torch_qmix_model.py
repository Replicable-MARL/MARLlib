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
from GRF.model.torch_cnn_updet import Transformer

torch, nn = try_import_torch()


class Torch_CNN_GRU_Model(TorchModelV2, nn.Module):
    """The GRU model only for QMIX."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, hidden_state_size=256, ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.hidden_state_size = hidden_state_size
        self.n_agents = model_config["n_agents"]

        # Build the Module from Conv + FC + GRU + 2xfc (action + value outs).
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=4,
                out_channels=8,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
            ),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=3),
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
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Linear(16 * 7 * 7, self.hidden_state_size)  # fully connected layer, output 10 classes
        self.rnn = nn.GRUCell(self.hidden_state_size, self.hidden_state_size)
        self.fc2 = nn.Linear(self.hidden_state_size, num_outputs)

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        return [
            self.fc1.weight.new(self.n_agents,
                                self.hidden_state_size).zero_().squeeze(0)
        ]

    @override(ModelV2)
    def forward(self, input_dict, hidden_state, seq_lens):
        b = input_dict["obs_flat"].shape[0]
        x = self.conv1(input_dict["obs_flat"].view(b, 42, 42, 4).permute(0, 3, 1, 2))
        x = self.conv2(x)
        x = nn.functional.relu(self.fc1(x.view(b, -1)))
        h_in = hidden_state[0].reshape(-1, self.hidden_state_size)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, [h]


class Torch_CNN_UPDeT_Model(TorchModelV2, nn.Module):
    """The UDPeT for QMIX."""

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
    ):
        self.conv_emb = 4 * 7 * 7
        self.trans_emb = emb
        self.heads = heads
        self.depth = depth
        self.n_agents = model_config["n_agents"]

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

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [
            self.action_branch.weight.new(self.n_agents, self.trans_emb).zero_().squeeze(0),
        ]
        return h

    @override(ModelV2)
    def forward(self, input_dict, hidden_state, seq_lens):
        b = input_dict["obs_flat"].shape[0]
        x = input_dict["obs_flat"].view(b, 42, 42, 4)
        x = x.permute(0, 3, 1, 2).reshape(-1, 1, 42, 42)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x.view(4 * b, -1))
        x = x.view(b, 4, self.trans_emb)

        # inputs = self._build_inputs_transformer(inputs)
        outputs, _ = self.transformer(x, hidden_state[0], None)

        # last dim for hidden state
        h = outputs[:, -1:, :]

        # record self._features
        self._features = torch.max(outputs[:, :-1, :], 1)[0]
        logits = self.action_branch(self._features)

        # Return masked logits.
        return logits, [torch.squeeze(h, 1)]

    def _build_inputs_transformer(self, inputs):
        pos = 4 - self.token_dim  # 5 for -1 6 for -2
        arranged_obs = torch.cat((inputs[:, pos:], inputs[:, :pos]), 1)
        reshaped_obs = arranged_obs.view(-1, 1 + (self.enemy_num - 1) + self.ally_num, self.token_dim)

        return reshaped_obs


def _get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size

