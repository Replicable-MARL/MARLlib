from ray.rllib.utils.torch_ops import FLOAT_MIN
import numpy as np
from typing import Dict, List, Any, Union
from gym.spaces import Box
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch, \
    TensorType
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from functools import reduce
import logging
import gym
import copy
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, \
    normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from marllib.marl.models.zoo.mlp.base_mlp import Base_MLP

torch, nn = try_import_torch()


class CC_MLP(Base_MLP):

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):

        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name, **kwargs)

        # encoder for centralized VF
        cc_layers = []
        if "state" not in self.full_obs_space.spaces:
            self.state_dim = self.full_obs_space["obs"].shape
            cc_input_dim = self.encoder_layer_dim[-1] * self.custom_config["num_agents"]

            self.cc_value_encoder = copy.deepcopy(self.value_encoder)
        else:
            self.state_dim = self.full_obs_space["state"].shape
            cc_input_dim = self.state_dim[0] + self.obs_size

            for cc_out_dim in self.encoder_layer_dim:
                cc_layers.append(
                    SlimFC(in_size=cc_input_dim,
                           out_size=cc_out_dim,
                           initializer=normc_initializer(1.0),
                           activation_fn=self.activation))
                cc_input_dim = cc_out_dim

            self.cc_value_encoder = nn.Sequential(*copy.deepcopy(cc_layers))

        if self.custom_config["opp_action_in_cc"]:
            if isinstance(self.custom_config["space_act"], Box):  # continuous
                cc_input_dim = cc_input_dim + num_outputs * (self.custom_config["num_agents"] - 1) // 2
            else:
                cc_input_dim = cc_input_dim + num_outputs * (self.custom_config["num_agents"] - 1)

        self.cc_value_branch = SlimFC(
            in_size=cc_input_dim,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        self.q_flag = False
        if self.custom_config["algorithm"] in ["coma"]:
            self.q_flag = True
            # self.value_branch = SlimFC(
            #     in_size=cc_input_dim,
            #     out_size=num_outputs,
            #     initializer=normc_initializer(0.01),
            #     activation_fn=None)
            self.cc_value_branch = SlimFC(
                in_size=cc_input_dim,
                out_size=num_outputs,
                initializer=normc_initializer(0.01),
                activation_fn=None)

    def central_value_function(self, state, opponent_actions=None) -> TensorType:
        assert self._features is not None, "must call forward() first"
        B = state.shape[0]

        if "conv_layer" in self.custom_config["model_arch_args"]:
            x = state.reshape(-1, self.state_dim[0], self.state_dim[1], self.state_dim_last).permute(0, 3, 1, 2)
            x = self.cc_value_encoder(x)
            x = torch.mean(x, (2, 3))
        else:
            x = self.cc_value_encoder(state)

        if opponent_actions is None:
            x = torch.cat([x.reshape(B, -1)], 1)
        else:
            if isinstance(self.custom_config["space_act"], Box):  # continuous
                opponent_actions_ls = [opponent_actions[:, i, :]
                                       for i in
                                       range(self.n_agents - 1)]
            else:
                opponent_actions_ls = [
                    torch.nn.functional.one_hot(opponent_actions[:, i].long(), self.num_outputs).float()
                    for i in
                    range(self.n_agents - 1)]

            x = torch.cat([x.reshape(B, -1)] + opponent_actions_ls, 1)

        if self.q_flag:
            return torch.reshape(self.cc_value_branch(x), [-1, self.num_outputs])
        else:
            return torch.reshape(self.cc_value_branch(x), [-1])
