from ray.rllib.utils.torch_ops import FLOAT_MIN
import numpy as np
from typing import Dict, List, Any, Union
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch, \
    TensorType
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.agents.qmix.mixers import QMixer, VDNMixer
import copy

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class Torch_CNN_GRU_Model_w_Mixer(TorchRNN, nn.Module):

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            fc_size=64,
            hidden_state_size=256,
            **kwargs,
    ):
        self.fc_size = fc_size
        self.hidden_state_size = hidden_state_size
        self.map_size = kwargs["map_size"]
        self.team_agent_num = 2
        self.all_agent_num = model_config["custom_model_config"]["agent_num"]

        self.mixer = model_config["custom_model_config"]["mixer"]
        self.mixer_embed_dim = model_config["custom_model_config"]["mixer_emb_dim"]

        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # Build the Module from Conv + FC + GRU + 2xfc (action + value outs).
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
        self.fc1 = nn.Linear(64, self.fc_size)
        self.fc2 = nn.Linear(4, self.fc_size)

        self.gru = nn.GRU(
            self.fc_size * 2, self.hidden_state_size, batch_first=True)
        self.action_branch = nn.Linear(self.hidden_state_size, num_outputs)
        self.value_branch = nn.Linear(self.hidden_state_size, 1)
        # Holds the current "base" output (before logits layer).

        self.conv1_cc = copy.deepcopy(self.conv1)
        self.conv2_cc = copy.deepcopy(self.conv2)
        self.fc1_cc = nn.Linear(64 * self.all_agent_num, self.hidden_state_size)
        # self.fc2_cc = nn.Linear(4 * self.n_agents, self.hidden_state_size)
        # self.value_branch_cc = nn.Linear(64 * self.all_agent_num + 4 * self.all_agent_num + 3 * 6, 1)
        self._features = None



        if self.mixer == "qmix":
            # use observation mean as state
            self.mixer_network = QMixer(self.team_agent_num, self.hidden_state_size + self.all_agent_num * 4, self.mixer_embed_dim)
        else: # "vdn"
            self.mixer_network = VDNMixer()

    @override(ModelV2)
    def get_initial_state(self):
        h = [
            self.fc1.weight.new(1, self.hidden_state_size).zero_().squeeze(0)
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

        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()

        flat_inputs_obs = input_dict["obs"]["obs"].float()
        self.time_major = self.model_config.get("_time_major", False)

        max_seq_len = flat_inputs_obs.shape[0] // seq_lens.shape[0]
        inputs_obs = add_time_dimension(
            flat_inputs_obs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )

        flat_inputs_status = input_dict["obs"]["status"].float()
        inputs_status = add_time_dimension(
            flat_inputs_status,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )

        output, new_state = self.forward_rnn([inputs_obs, inputs_status], state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])

        return output, new_state

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        # Extract the available actions tensor from the observation.

        # Compute the unmasked logits.
        obs, status = inputs
        x = self.conv1(obs.view(-1, obs.shape[2], obs.shape[3], obs.shape[4]).permute(0, 3, 1, 2))
        x = self.conv2(x)
        x_obs = nn.functional.relu(self.fc1(x.view(obs.shape[0], obs.shape[1], -1)))
        x_status = nn.functional.relu(self.fc2(status))
        x = torch.cat((x_obs, x_status), 2)
        self._features, h = self.gru(x, torch.unsqueeze(state[0], 0))
        logits = self.action_branch(self._features)

        # Return masked logits.
        return logits, [torch.squeeze(h, 0)]

    def mixing_value(self, all_agents_vf, obs, opponent_obs):
        # compatiable with rllib qmix mixer
        # reshape to raw shape
        obs_self = obs[:, :-4].view(-1, self.map_size, self.map_size, 5).unsqueeze(1)
        status_self = obs[:, -4:].unsqueeze(1)
        obs_opponent = opponent_obs[:, :, :-4].view(-1, self.all_agent_num - 1, self.map_size, self.map_size, 5)
        status_opponent = opponent_obs[:, :, -4:]

        obs_agg = torch.cat((obs_self, obs_opponent), 1).reshape(-1, self.map_size, self.map_size, 5).permute(0, 3, 1,
                                                                                                              2)
        status_agg = torch.cat((status_self, status_opponent), 1).reshape(-1, self.all_agent_num * 4)

        # obs
        x = self.conv1_cc(obs_agg)
        x = self.conv2_cc(x)
        x = x.reshape(x.shape[0], -1).view(x.shape[0] // self.all_agent_num, -1)
        x = self.fc1_cc(x)

        # status
        x_s = status_agg.view(status_agg.shape[0], -1)

        # concat all global info
        state = torch.cat([x, x_s, ], 1)
        all_agents_vf = all_agents_vf.view(-1, 1, self.team_agent_num)
        v_tot = self.mixer_network(all_agents_vf, state)

        # shape to [B]
        return v_tot.flatten(start_dim=0)


class Torch_CNN_LSTM_Model_w_Mixer(TorchRNN, nn.Module):

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            fc_size=64,
            hidden_state_size=256,
            **kwargs,
    ):
        # self.obs_size = obs_space.shape[0]
        self.fc_size = fc_size
        self.hidden_state_size = hidden_state_size
        self.map_size = kwargs["map_size"]
        self.team_agent_num = 2
        self.all_agent_num = model_config["custom_model_config"]["agent_num"]

        self.mixer = model_config["custom_model_config"]["mixer"]
        self.mixer_embed_dim = model_config["custom_model_config"]["mixer_emb_dim"]

        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # Build the Module from Conv + FC + GRU + 2xfc (action + value outs).
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
        self.fc1 = nn.Linear(64, self.fc_size)
        self.fc2 = nn.Linear(4, self.fc_size)

        self.lstm = nn.LSTM(
            self.fc_size * 2, self.hidden_state_size, batch_first=True)
        self.action_branch = nn.Linear(self.hidden_state_size, num_outputs)
        self.value_branch = nn.Linear(self.hidden_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self.conv1_cc = copy.deepcopy(self.conv1)
        self.conv2_cc = copy.deepcopy(self.conv2)
        self.fc1_cc = nn.Linear(64 * self.all_agent_num, self.hidden_state_size)
        # self.fc2_cc = nn.Linear(4 * self.n_agents, self.hidden_state_size)
        # self.value_branch_cc = nn.Linear(64 * self.all_agent_num + 4 * self.all_agent_num + 3 * 6, 1)
        self._features = None

        if self.mixer == "qmix":
            # use observation mean as state
            self.mixer_network = QMixer(self.team_agent_num, self.hidden_state_size + self.all_agent_num * 4, self.mixer_embed_dim)
        else: # "vdn"
            self.mixer_network = VDNMixer()

    @override(ModelV2)
    def get_initial_state(self):
        h = [
            self.fc1.weight.new(1, self.hidden_state_size).zero_().squeeze(0),
            self.fc1.weight.new(1, self.hidden_state_size).zero_().squeeze(0)
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

        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()

        flat_inputs_obs = input_dict["obs"]["obs"].float()
        self.time_major = self.model_config.get("_time_major", False)

        max_seq_len = flat_inputs_obs.shape[0] // seq_lens.shape[0]
        inputs_obs = add_time_dimension(
            flat_inputs_obs,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )

        flat_inputs_status = input_dict["obs"]["status"].float()
        inputs_status = add_time_dimension(
            flat_inputs_status,
            max_seq_len=max_seq_len,
            framework="torch",
            time_major=self.time_major,
        )

        output, new_state = self.forward_rnn([inputs_obs, inputs_status], state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])

        return output, new_state

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        # Extract the available actions tensor from the observation.

        # Compute the unmasked logits.
        obs, status = inputs
        x = self.conv1(obs.view(-1, obs.shape[2], obs.shape[3], obs.shape[4]).permute(0, 3, 1, 2))
        x = self.conv2(x)
        x_obs = nn.functional.relu(self.fc1(x.view(obs.shape[0], obs.shape[1], -1)))
        x_status = nn.functional.relu(self.fc2(status))
        x = torch.cat((x_obs, x_status), 2)

        self._features, [h, c] = self.lstm(
            x, [torch.unsqueeze(state[0], 0),
                torch.unsqueeze(state[1], 0)])
        logits = self.action_branch(self._features)

        # Return masked logits.
        return logits, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    def mixing_value(self, all_agents_vf, obs, opponent_obs):
        # compatiable with rllib qmix mixer
        # reshape to raw shape
        obs_self = obs[:, :-4].view(-1, self.map_size, self.map_size, 5).unsqueeze(1)
        status_self = obs[:, -4:].unsqueeze(1)
        obs_opponent = opponent_obs[:, :, :-4].view(-1, self.all_agent_num - 1, self.map_size, self.map_size, 5)
        status_opponent = opponent_obs[:, :, -4:]

        obs_agg = torch.cat((obs_self, obs_opponent), 1).reshape(-1, self.map_size, self.map_size, 5).permute(0, 3, 1,
                                                                                                              2)
        status_agg = torch.cat((status_self, status_opponent), 1).reshape(-1, self.all_agent_num * 4)

        # obs
        x = self.conv1_cc(obs_agg)
        x = self.conv2_cc(x)
        x = x.reshape(x.shape[0], -1).view(x.shape[0] // self.all_agent_num, -1)
        x = self.fc1_cc(x)

        # status
        x_s = status_agg.view(status_agg.shape[0], -1)

        # concat all global info
        state = torch.cat([x, x_s, ], 1)
        all_agents_vf = all_agents_vf.view(-1, 1, self.team_agent_num)
        v_tot = self.mixer_network(all_agents_vf, state)

        # shape to [B]
        return v_tot.flatten(start_dim=0)