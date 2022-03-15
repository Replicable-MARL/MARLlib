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
from SMAC.model.torch_mask_updet import Transformer

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class Torch_ActionMask_GRU_Model_w_Mixer(TorchRNN, nn.Module):

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
        full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.obs_size = full_obs_space['obs'].shape[0]
        self.state_size = full_obs_space['state'].shape[0]

        self.fc_size = fc_size
        self.hidden_state_size = hidden_state_size
        self.mixer = model_config["custom_model_config"]["mixer"]
        self.mixer_embed_dim = model_config["custom_model_config"]["mixer_emb_dim"]
        self.n_agents = model_config["custom_model_config"]["ally_num"]

        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # Build the Module from fc + GRU + 2xfc (action + value outs).
        self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        self.gru = nn.GRU(
            self.fc_size, self.hidden_state_size, batch_first=True)
        self.action_branch = nn.Linear(self.hidden_state_size, num_outputs)
        self.value_branch = nn.Linear(self.hidden_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

        if self.mixer == "qmix":
            self.mixer_network = QMixer(self.n_agents, self.state_size, self.mixer_embed_dim)
        else: # "vdn"
            self.mixer_network = VDNMixer()


    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.fc1.weight.new(1, self.hidden_state_size).zero_().squeeze(0),
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
        flat_inputs = input_dict["obs"]["obs"].float()
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

        # Convert action_mask into a [0.0 || -inf]-type mask.
        action_mask = input_dict["obs"]["action_mask"]
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)

        # for r2d2 output is dim 256
        if output.shape[1] != self.action_space.n:
            masked_output = [output, inf_mask]
        else:
            masked_output = output + inf_mask

        return masked_output, new_state

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        # Extract the available actions tensor from the observation.

        # Compute the unmasked logits.
        x = nn.functional.relu(self.fc1(inputs))
        self._features, h = self.gru(x, torch.unsqueeze(state[0], 0))
        logits = self.action_branch(self._features)

        # Return masked logits.
        return logits, [torch.squeeze(h, 0)]

    def mixing_value(self, all_agents_vf, state):
        # compatiable with rllib qmix mixer
        all_agents_vf = all_agents_vf.view(-1, 1, self.n_agents)
        v_tot = self.mixer_network(all_agents_vf, state)

        # shape to [B]
        return v_tot.flatten(start_dim=0)


class Torch_ActionMask_LSTM_Model_w_Mixer(TorchRNN, nn.Module):

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
        full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.obs_size = full_obs_space['obs'].shape[0]
        self.state_size = full_obs_space['state'].shape[0]

        self.fc_size = fc_size
        self.lstm_state_size = hidden_state_size
        self.mixer = model_config["custom_model_config"]["mixer"]
        self.mixer_embed_dim = model_config["custom_model_config"]["mixer_emb_dim"]
        self.n_agents = model_config["custom_model_config"]["ally_num"]

        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # Build the Module from fc + GRU + 2xfc (action + value outs).
        self.fc1 = nn.Linear(self.obs_size, self.fc_size)
        self.lstm = nn.LSTM(
            self.fc_size, self.lstm_state_size, batch_first=True)
        self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
        self.value_branch = nn.Linear(self.lstm_state_size, 1)
        # Holds the current "base" output (before logits layer).
        self._features = None

        if self.mixer == "qmix":
            self.mixer_network = QMixer(self.n_agents, self.state_size, self.mixer_embed_dim)
        else: # "vdn"
            self.mixer_network = VDNMixer()


    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
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
        flat_inputs = input_dict["obs"]["obs"].float()
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

        # Convert action_mask into a [0.0 || -inf]-type mask.
        action_mask = input_dict["obs"]["action_mask"]
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)

        # for r2d2 output is dim 256
        if output.shape[1] != self.action_space.n:
            masked_output = [output, inf_mask]
        else:
            masked_output = output + inf_mask

        return masked_output, new_state

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

    def mixing_value(self, all_agents_vf, state):
        # compatiable with rllib qmix mixer
        all_agents_vf = all_agents_vf.view(-1, 1, self.n_agents)
        v_tot = self.mixer_network(all_agents_vf, state)

        # shape to [B]
        return v_tot.flatten(start_dim=0)


class Torch_ActionMask_Transformer_Model_w_Mixer(TorchRNN, nn.Module):

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
        full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.obs_size = full_obs_space['obs'].shape[0]
        self.state_size = full_obs_space['state'].shape[0]
        self.emb = emb
        self.heads = heads
        self.depth = depth
        self.token_dim = model_config["custom_model_config"]["token_dim"]
        self.ally_num = model_config["custom_model_config"]["ally_num"]
        self.enemy_num = model_config["custom_model_config"]["enemy_num"]
        self.mixer = model_config["custom_model_config"]["mixer"]
        self.mixer_embed_dim = model_config["custom_model_config"]["mixer_emb_dim"]
        self.n_agents = model_config["custom_model_config"]["ally_num"]

        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        self._features = None

        # Build the Module from Transformer / regard as RNN
        self.transformer = Transformer(self.token_dim, self.emb, self.heads, self.depth, self.emb)
        self.action_branch = nn.Linear(self.emb, 6)
        # Critic using token output concat exclude hidden state
        self.value_branch = nn.Linear(self.emb * (self.ally_num + self.enemy_num), 1)

        if self.mixer == "qmix":
            self.mixer_network = QMixer(self.n_agents, self.state_size, self.mixer_embed_dim)
        else: # "vdn"
            self.mixer_network = VDNMixer()

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.action_branch.weight.new(1, self.emb).zero_().squeeze(0),
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
        flat_inputs = input_dict["obs"]["obs"].float()
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

        # Convert action_mask into a [0.0 || -inf]-type mask.
        action_mask = input_dict["obs"]["action_mask"]
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_output = output + inf_mask

        return masked_output, state

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        # Extract the available actions tensor from the observation.
        inputs = self._build_inputs_transformer(inputs)
        outputs, _ = self.transformer(inputs, state[0], None)

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
        arranged_obs = torch.cat((inputs[:, :, pos:], inputs[:, :, :pos]), 2)
        reshaped_obs = arranged_obs.view(-1, 1 + (self.enemy_num - 1) + self.ally_num, self.token_dim)

        return reshaped_obs

    def mixing_value(self, all_agents_vf, state):
        # compatiable with rllib qmix mixer
        all_agents_vf = all_agents_vf.view(-1, 1, self.n_agents)
        v_tot = self.mixer_network(all_agents_vf, state)

        # shape to [B]
        return v_tot.flatten(start_dim=0)

