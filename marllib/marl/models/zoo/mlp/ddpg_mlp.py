# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Dict, List
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch, \
    TensorType
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from marllib.marl.models.zoo.mixer import QMixer, VDNMixer
from marllib.marl.models.zoo.encoder.base_encoder import BaseEncoder

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

class DDPGSeriesMLP(TorchModelV2, nn.Module):
    """
    IDDPG/MADDPG/FACMAC agent arch in one model
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

        self.custom_config = model_config["custom_model_config"]
        self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        self.n_agents = self.custom_config["num_agents"]
        self.activation = model_config.get("fcnet_activation")

        # encoder
        self.encoder = BaseEncoder(model_config, self.full_obs_space)

        if self.custom_config["algorithm"] in ["maddpg"]:
            all_action_dim = self.custom_config["space_act"].shape[0] * self.custom_config["num_agents"]
            if self.custom_config["global_state_flag"]:
                self.state_encoder = nn.Linear(self.full_obs_space["state"].shape[0],
                                               self.encoder.output_dim)
            if "q" in name:
                if self.custom_config["global_state_flag"]:
                    input_dim = self.encoder.output_dim * 2 + all_action_dim
                else:
                    input_dim = self.encoder.output_dim * self.custom_config["num_agents"] + all_action_dim
            else:
                input_dim = self.encoder.output_dim

        else:  # no centralized critic -> iddpg
            if "q" in name:
                input_dim = self.encoder.output_dim + self.custom_config["space_act"].shape[0]
            else:
                input_dim = self.encoder.output_dim

        self.hidden_state_size = self.custom_config["model_arch_args"]["hidden_state_size"]
        self.input_dim = input_dim

        self.mlp = nn.Linear(input_dim, self.hidden_state_size)

        # action branch and value branch
        self.out_branch = SlimFC(
            in_size=self.hidden_state_size,
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        if self.custom_config["algorithm"] in ["facmac"]:
            # mixer:
            if self.custom_config["global_state_flag"]:
                state_dim = self.custom_config["space_obs"]["state"].shape
            else:
                state_dim = self.custom_config["space_obs"]["obs"].shape + (self.custom_config["num_agents"],)

            mixer_arch = model_config["custom_model_config"]["model_arch_args"]["mixer_arch"]
            if mixer_arch == "qmix":
                self.mixer = QMixer(self.custom_config, state_dim)
            elif mixer_arch == "vdn":
                self.mixer = VDNMixer()

        # Holds the current "base" output (before logits layer).
        self._features = None

        # record the custom config
        self.n_agents = self.custom_config["num_agents"]
        self.q_flag = False

    @override(ModelV2)
    def get_initial_state(self): # not used
        # Place hidden states on same device as model.
        return self.out_branch._model._modules["0"].weight.new(1, self.hidden_state_size).zero_().squeeze(0),

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
            x = self.encoder(obs_inputs)

            if action_inputs is not None:
                x = torch.cat((x, action_inputs), -1)

        x = nn.functional.relu(x)

        self._features = self.mlp(x)
        logits = self.out_branch(self._features)

        return logits, state

    def mixing_value(self, all_agents_q, state):
        # compatiable with rllib qmix mixer
        all_agents_q = all_agents_q.view(-1, 1, self.n_agents)
        q_tot = self.mixer(all_agents_q, state)

        # shape to [B]
        return q_tot.flatten(start_dim=0)