from SMAC.model.torch_mask_updet import *

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class Torch_ActionMask_Transformer_CentralizedCritic_Model(TorchRNN, nn.Module):
    """Multi-agent model that implements a centralized VF."""

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
        self.emb = emb
        self.heads = heads
        self.depth = depth
        self.token_dim = model_config["custom_model_config"]["token_dim"]
        self.ally_num = model_config["custom_model_config"]["ally_num"]
        self.enemy_num = model_config["custom_model_config"]["enemy_num"]

        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # Build the Module from Transformer / regard as RNN
        self.transformer = Transformer(self.token_dim, self.emb, self.heads, self.depth, self.emb)
        self.action_branch = nn.Linear(self.emb, 6)
        # Critic using token output concat exclude hidden state
        self.value_branch = nn.Linear(self.emb * (self.ally_num + self.enemy_num), 1)

        # Holds the current "base" output (before logits layer).
        self._features = None

        # Central VF maps (obs, opp_obs, opp_act) -> vf_pred
        obs_dim = kwargs['self_obs_dim']
        state_dim = kwargs['state_dim']
        input_size = obs_dim + state_dim + num_outputs * (self.ally_num - 1)  # obs + opp_obs + opp_act
        self.central_vf = nn.Sequential(
            SlimFC(input_size, 32, activation_fn=nn.Tanh),
            SlimFC(32, 1),
        )

        self.coma_flag = False
        if "coma" in model_config["custom_model_config"]:
            self.coma_flag = True
            self.central_vf = nn.Sequential(
                SlimFC(input_size, 16, activation_fn=nn.Tanh),
                SlimFC(16, num_outputs),
            )

    @override(TorchRNN)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.action_branch.weight.new(1, self.emb).zero_().squeeze(0),
        ]
        return h

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
        # currently we only support battles with marines (e.g. 3m 8m 5m_vs_6m)
        # you can implement your own with any other agent type.
        arranged_obs = torch.cat((inputs[:, :, -1:], inputs[:, :, :-1]), 2)
        reshaped_obs = arranged_obs.view(-1, 1 + (self.enemy_num - 1) + self.ally_num, self.token_dim)

        return reshaped_obs

    # here we use individual observation + global state as input of critic
    def central_value_function(self, obs, state, opponent_actions):
        opponent_actions_one_hot = [
            torch.nn.functional.one_hot(opponent_actions[:, i].long(), self.num_outputs).float()
            for i in
            range(opponent_actions.shape[1])]
        input_ = torch.cat([obs, state] + opponent_actions_one_hot, 1)
        if self.coma_flag:
            return torch.reshape(self.central_vf(input_), [-1, self.num_outputs])
        else:
            return torch.reshape(self.central_vf(input_), [-1])

    @override(TorchRNN)  # not used
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

