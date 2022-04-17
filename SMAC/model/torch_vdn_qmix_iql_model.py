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
from SMAC.model.torch_mask_updet import Transformer
from torch.optim import Adam
torch, nn = try_import_torch()


class GRUModel(TorchModelV2, nn.Module):
    """The default GRU model for QMIX."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        self.obs_size = _get_size(obs_space)
        self.rnn_hidden_dim = model_config["lstm_cell_size"]
        self.fc1 = nn.Linear(self.obs_size, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, num_outputs)
        self.n_agents = model_config["n_agents"]

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        return [
            self.fc1.weight.new(self.n_agents,
                                self.rnn_hidden_dim).zero_().squeeze(0)
        ]

    @override(ModelV2)
    def forward(self, input_dict, hidden_state, seq_lens):
        x = nn.functional.relu(self.fc1(input_dict["obs_flat"].float()))
        h_in = hidden_state[0].reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, [h]


class UPDeTModel(TorchModelV2, nn.Module):
    """The UDPeT for QMIX."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name,
                 emb=32,
                 heads=3,
                 depth=2, ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)
        self.obs_size = _get_size(obs_space)
        self.emb = emb
        self.heads = heads
        self.depth = depth
        self.token_dim = model_config["custom_model_config"]["token_dim"]
        self.ally_num = model_config["custom_model_config"]["ally_num"]
        self.enemy_num = model_config["custom_model_config"]["enemy_num"]
        self.transformer = Transformer(self.token_dim, self.emb, self.heads, self.depth, self.emb)
        self.action_branch = nn.Linear(self.emb, 6)
        self.n_agents = model_config["n_agents"]

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        h = [
            self.action_branch.weight.new(self.n_agents, self.emb).zero_().squeeze(0),
        ]
        return h

    @override(ModelV2)
    def forward(self, input_dict, hidden_state, seq_lens):
        # Extract the available actions tensor from the observation.
        inputs = self._build_inputs_transformer(input_dict["obs_flat"].float())
        outputs, _ = self.transformer(inputs, hidden_state[0], None)

        # record self._features
        self._features = outputs[:, :-1, :]
        # last dim for hidden state
        h = outputs[:, -1:, :]

        # Compute the unmasked logits.
        q_basic_actions = self.action_branch(outputs[:, 0, :])

        q_enemies_list = []
        # each enemy has an output Q
        for i in range(self.enemy_num):
            q_enemy = self.action_branch(outputs[:, 1 + i, :])
            q_enemy_mean = torch.mean(q_enemy, 1, True)
            q_enemies_list.append(q_enemy_mean)

        # concat enemy Q over all enemies
        q_enemies = torch.stack(q_enemies_list, dim=1).squeeze(dim=2)

        # concat basic action Q with enemy attack Q
        logits = torch.cat((q_basic_actions, q_enemies), 1)

        # Return masked logits.
        return logits, [torch.squeeze(h, 1)]

    def _build_inputs_transformer(self, inputs):
        pos = 4 - self.token_dim  # 5 for -1 6 for -2
        arranged_obs = torch.cat((inputs[:, pos:], inputs[:, :pos]), 1)
        reshaped_obs = arranged_obs.view(-1, 1 + (self.enemy_num - 1) + self.ally_num, self.token_dim)

        return reshaped_obs


def _get_size(obs_space):
    return get_preprocessor(obs_space)(obs_space).size



