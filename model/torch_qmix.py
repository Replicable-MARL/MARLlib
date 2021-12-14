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
from model.torch_mask_updet import Transformer

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


class Customized_QMixTorchPolicy(QMixTorchPolicy):

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config)
        config["model"]["n_agents"] = self.n_agents

        agent_obs_space = obs_space.original_space.spaces[0]
        if isinstance(agent_obs_space, Dict):
            space_keys = set(agent_obs_space.spaces.keys())
            if "obs" not in space_keys:
                raise ValueError(
                    "Dict obs space must have subspace labeled `obs`")
            self.obs_size = _get_size(agent_obs_space.spaces["obs"])
            if "action_mask" in space_keys:
                mask_shape = tuple(agent_obs_space.spaces["action_mask"].shape)
                if mask_shape != (self.n_actions,):
                    raise ValueError(
                        "Action mask shape must be {}, got {}".format(
                            (self.n_actions,), mask_shape))
                self.has_action_mask = True
            if ENV_STATE in space_keys:
                self.env_global_state_shape = _get_size(
                    agent_obs_space.spaces[ENV_STATE])
                self.has_env_global_state = True
            else:
                self.env_global_state_shape = (self.obs_size, self.n_agents)
            # The real agent obs space is nested inside the dict
            config["model"]["full_obs_space"] = agent_obs_space
            agent_obs_space = agent_obs_space.spaces["obs"]
        else:
            self.obs_size = _get_size(agent_obs_space)
            self.env_global_state_shape = (self.obs_size, self.n_agents)

        neural_arch = config["model"]["custom_model_config"]["neural_arch"]
        self.model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space.spaces[0],
            self.n_actions,
            config["model"],
            framework="torch",
            name="model",
            default_model=GRUModel if neural_arch == "GRU" else UPDeTModel).to(self.device)

        self.target_model = ModelCatalog.get_model_v2(
            agent_obs_space,
            action_space.spaces[0],
            self.n_actions,
            config["model"],
            framework="torch",
            name="target_model",
            default_model=GRUModel if neural_arch == "GRU" else UPDeTModel).to(self.device)

        self.exploration = self._create_exploration()

        # Setup the mixer network.
        if config["mixer"] is None:
            self.mixer = None
            self.target_mixer = None
        elif config["mixer"] == "qmix":
            self.mixer = QMixer(self.n_agents, self.env_global_state_shape,
                                config["mixing_embed_dim"]).to(self.device)
            self.target_mixer = QMixer(
                self.n_agents, self.env_global_state_shape,
                config["mixing_embed_dim"]).to(self.device)
        elif config["mixer"] == "vdn":
            self.mixer = VDNMixer().to(self.device)
            self.target_mixer = VDNMixer().to(self.device)
        else:
            raise ValueError("Unknown mixer type {}".format(config["mixer"]))

        self.cur_epsilon = 1.0
        self.update_target()  # initial sync

        # Setup optimizer
        self.params = list(self.model.parameters())
        if self.mixer:
            self.params += list(self.mixer.parameters())
        self.loss = QMixLoss(self.model, self.target_model, self.mixer,
                             self.target_mixer, self.n_agents, self.n_actions,
                             self.config["double_q"], self.config["gamma"])
        from torch.optim import RMSprop
        self.optimiser = RMSprop(
            params=self.params,
            lr=config["lr"],
            alpha=config["optim_alpha"],
            eps=config["optim_eps"])


QMixTrainer = GenericOffPolicyTrainer.with_updates(
    name="QMIX",
    default_config=DEFAULT_CONFIG,
    default_policy=Customized_QMixTorchPolicy,
    get_policy_class=None,
    execution_plan=execution_plan)
