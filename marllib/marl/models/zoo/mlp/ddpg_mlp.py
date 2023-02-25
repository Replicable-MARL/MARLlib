from typing import Dict, List
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch, \
    TensorType
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.misc import SlimFC, SlimConv2d, normc_initializer
from marllib.marl.models.zoo.mixer import QMixer, VDNMixer

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class DDPG_MLP(TorchModelV2, nn.Module):
    """
    DDOG/MADDPG/FACMAC agent arch in one model
    """

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)

        # judge the model arch
        self.custom_config = model_config["custom_model_config"]
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.n_agents = self.custom_config["num_agents"]
        self.activation = model_config.get("fcnet_activation")

        if self.custom_config["model_arch_args"]["core_arch"] != "mlp":
            raise ValueError()
        # encoder
        layers = []
        if "fc_layer" in self.custom_config["model_arch_args"]:
            if "encode_layer" in self.custom_config["model_arch_args"]:
                encode_layer = self.custom_config["model_arch_args"]["encode_layer"]
                encoder_layer_dim = encode_layer.split("-")
                encoder_layer_dim = [int(i) for i in encoder_layer_dim]
            else:  # default config
                encoder_layer_dim = []
                for i in range(self.custom_config["model_arch_args"]["fc_layer"]):
                    out_dim = self.custom_config["model_arch_args"]["out_dim_fc_{}".format(i)]
                    encoder_layer_dim.append(out_dim)

            self.encoder_layer_dim = encoder_layer_dim
            self.obs_size = self.full_obs_space["obs"].shape[0]
            input_dim = self.obs_size
            for out_dim in self.encoder_layer_dim:
                layers.append(
                    SlimFC(in_size=input_dim,
                           out_size=out_dim,
                           initializer=normc_initializer(1.0),
                           activation_fn=self.activation))
                input_dim = out_dim
        elif "conv_layer" in self.custom_config["model_arch_args"]:
            self.obs_size = self.full_obs_space['obs'].shape
            input_dim = self.obs_size[2]
            for i in range(self.custom_config["model_arch_args"]["conv_layer"]):
                layers.append(
                    SlimConv2d(
                        in_channels=input_dim,
                        out_channels=self.custom_config["model_arch_args"]["out_channel_layer_{}".format(i)],
                        kernel=self.custom_config["model_arch_args"]["kernel_size_layer_{}".format(i)],
                        stride=self.custom_config["model_arch_args"]["stride_layer_{}".format(i)],
                        padding=self.custom_config["model_arch_args"]["padding_layer_{}".format(i)],
                        activation_fn=self.activation
                    )
                )
                pool_f = nn.MaxPool2d(
                    kernel_size=self.custom_config["model_arch_args"]["pool_size_layer_{}".format(i)])
                layers.append(pool_f)

                input_dim = self.custom_config["model_arch_args"]["out_channel_layer_{}".format(i)]

        else:
            raise ValueError()

        self.encoder = nn.Sequential(
            *layers
        )

        if self.custom_config["algorithm"] in ["maddpg"]:
            all_action_dim = self.custom_config["space_act"].shape[0] * self.custom_config["num_agents"]
            if self.custom_config["global_state_flag"]:
                self.state_encoder = nn.Linear(self.full_obs_space["state"].shape[0],
                                               input_dim)
            if "q" in name:
                if self.custom_config["global_state_flag"]:
                    input_dim = input_dim * 2 + all_action_dim
                else:
                    input_dim = input_dim * self.custom_config["num_agents"] + all_action_dim

        else:  # no centralized critic -> iddpg
            if "q" in name:
                input_dim = input_dim + self.custom_config["space_act"].shape[0]

        # core rnn
        self.hidden_state_size = self.custom_config["model_arch_args"]["hidden_state_size"]
        self.input_dim = input_dim

        self.mlp = nn.Linear(input_dim, self.hidden_state_size)

        # action branch and value branch
        self.action_branch = nn.Linear(self.hidden_state_size, num_outputs)
        self.value_branch = nn.Linear(self.hidden_state_size, 1)

        if self.custom_config["algorithm"] in ["facmac"]:
            # mixer:
            if self.custom_config["global_state_flag"]:
                state_dim = self.custom_config["space_obs"]["state"].shape
            else:
                state_dim = self.custom_config["space_obs"]["obs"].shape + (self.custom_config["num_agents"],)
            if self.custom_config["algo_args"]["mixer"] == "qmix":
                self.mixer = QMixer(self.custom_config, state_dim)
            elif self.custom_config["algo_args"]["mixer"] == "vdn":
                self.mixer = VDNMixer()
            else:
                raise ValueError("Unknown mixer type {}".format(self.custom_config["algo_args"]["mixer"]))

        # Holds the current "base" output (before logits layer).
        self._features = None

        # record the custom config
        self.n_agents = self.custom_config["num_agents"]
        self.q_flag = False

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        return self.value_branch.weight.new(1, self.hidden_state_size).zero_().squeeze(0),

    @override(ModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """
        Adds time dimension to batch before sending inputs to forward_rnn()
        """

        obs_inputs = input_dict["obs"]["obs"].float()

        if "state" in input_dict:
            state_inputs = input_dict["state"].float()
        else:
            state_inputs = None

        if "opponent_actions" in input_dict:
            opp_action_inputs = input_dict["opponent_actions"].float()
        else:
            opp_action_inputs = None

        if "actions" in input_dict:
            action_inputs = input_dict["actions"].float()
        else:
            action_inputs = None

        output, new_state = self.forward_mlp(obs_inputs, action_inputs, state_inputs, opp_action_inputs, state,
                                             seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])

        return output, new_state

    def forward_mlp(self, obs_inputs, action_inputs, state_inputs, opp_action_inputs, state, seq_lens):
        # Extract the available actions tensor from the observation.
        # Compute the unmasked logits.
        if self.custom_config["algorithm"] in ["maddpg"]:
            # CNN not support in MADDPG
            if action_inputs is not None:
                B = obs_inputs.shape[0]
                obs_x = self.encoder(obs_inputs)
                if self.custom_config["global_state_flag"]:
                    state_x = self.state_encoder(state_inputs)
                    x = torch.cat((obs_x, state_x, action_inputs, opp_action_inputs.reshape(B, -1)), -1)
                else:
                    state_x_ls = []
                    for i in range(self.n_agents):
                        state_x = self.encoder(state_inputs[:, i])
                        state_x_ls.append(state_x)
                    state_x = torch.cat(state_x_ls, -1)
                    x = torch.cat((state_x, action_inputs, opp_action_inputs.reshape(B, -1)), -1)

            else:
                x = self.encoder(obs_inputs)

        else:
            if "conv_layer" in self.custom_config["model_arch_args"]:
                x = obs_inputs.reshape(-1, obs_inputs.shape[2], obs_inputs.shape[3], obs_inputs.shape[4]).permute(0, 3,
                                                                                                                  2)
                x = self.encoder(x)
                x = torch.mean(x, (2, 3))
                x = x.reshape(obs_inputs.shape[0], obs_inputs.shape[1], -1)
            else:
                x = self.encoder(obs_inputs)

            if action_inputs is not None:
                x = torch.cat((x, action_inputs), -1)

        x = nn.functional.relu(x)

        self._features = self.mlp(x)
        logits = self.action_branch(self._features)

        return logits, state

    def mixing_value(self, all_agents_q, state):
        # compatiable with rllib qmix mixer
        all_agents_q = all_agents_q.view(-1, 1, self.n_agents)
        q_tot = self.mixer(all_agents_q, state)

        # shape to [B]
        return q_tot.flatten(start_dim=0)