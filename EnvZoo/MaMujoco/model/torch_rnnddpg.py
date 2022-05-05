import numpy as np
import gym
import copy
from typing import List, Dict, Optional
from gym.spaces import Box

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.agents.ddpg.ddpg_torch_model import DDPGTorchModel
from ray.rllib.utils import override, force_list
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.catalog import ModelCatalog


torch, nn = try_import_torch()


class RNNDDPGTorchModel(DDPGTorchModel):
    """Extension of standard DDPGTorchModel for RNNDDPG.

    Data flow:
        obs -> forward() -> model_out
        model_out -> get_policy_output() -> pi(s)
        model_out, actions -> get_q_values() -> Q(s, a)
        model_out, actions -> get_twin_q_values() -> Q_twin(s, a)

    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass."""

    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            twin_q: bool = False,
            add_layer_norm: bool = False,
            policy_model_config: ModelConfigDict = None,
            q_model_config: ModelConfigDict = None):

        super().__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
            twin_q=twin_q,
            add_layer_norm=add_layer_norm)

        self.use_prev_action = (model_config["lstm_use_prev_action"]
                                or policy_model_config["lstm_use_prev_action"]
                                or q_model_config["lstm_use_prev_action"])

        self.use_prev_reward = (model_config["lstm_use_prev_reward"]
                                or policy_model_config["lstm_use_prev_reward"]
                                or q_model_config["lstm_use_prev_reward"])
        if self.use_prev_action:
            self.view_requirements[SampleBatch.PREV_ACTIONS] = \
                ViewRequirement(SampleBatch.ACTIONS, space=self.action_space,
                                shift=-1)
        if self.use_prev_reward:
            self.view_requirements[SampleBatch.PREV_REWARDS] = \
                ViewRequirement(SampleBatch.REWARDS, shift=-1)

        self.action_dim = np.product(action_space.shape)
        self.discrete = False
        action_outs = self.action_dim
        q_outs = 1

        # Build the policy network.
        self.policy_model = self.build_policy_model(
            self.obs_space, action_outs, policy_model_config, "policy_model")

        class _Lambda(nn.Module):
            def __init__(self_):
                super().__init__()
                low_action = nn.Parameter(
                    torch.from_numpy(self.action_space.low).float())
                low_action.requires_grad = False
                self_.register_parameter("low_action", low_action)
                action_range = nn.Parameter(
                    torch.from_numpy(self.action_space.high -
                                     self.action_space.low).float())
                action_range.requires_grad = False
                self_.register_parameter("action_range", action_range)

            def forward(self_, x):
                sigmoid_out = nn.Sigmoid()(2.0 * x)
                squashed = self_.action_range * sigmoid_out + self_.low_action
                return squashed

        # Only squash if we have bounded actions.
        if self.bounded:
            self.policy_model.add_module("action_out_squashed", _Lambda())

        # Build the Q-network(s).
        self.q_model = self.build_q_model(self.obs_space, self.action_space,
                                          q_outs, q_model_config, "q")

        # Build the Q-network(s).
        self.q_model = self.build_q_model(self.obs_space, self.action_space,
                                          q_outs, q_model_config, "q")
        if twin_q:
            self.twin_q_model = self.build_q_model(self.obs_space,
                                                   self.action_space, q_outs,
                                                   q_model_config, "twin_q")
        else:
            self.twin_q_model = None

    # TODO 
    def build_policy_model(self, obs_space, num_outputs, policy_model_config,
                           name):
        """Builds the policy model used by this DDPG.

        Override this method in a sub-class of DDPGTFModel to implement your
        own policy net. Alternatively, simply set `custom_model` within the
        top level DDPG `policy_model` config key to make this default
        implementation of `build_policy_model` use your custom policy network.

        Returns:
            TorchModelV2: The TorchModelV2 policy sub-model.
        """
        model = ModelCatalog.get_model_v2(
            obs_space,
            self.action_space,
            num_outputs,
            policy_model_config,
            framework="torch",
            name=name)
        return model

    # TODO 
    def build_q_model(self, obs_space, action_space, num_outputs,
                      q_model_config, name):
        """Builds one of the (twin) Q-nets used by this DDPG.

        Override this method in a sub-class of DDPGTFModel to implement your
        own Q-nets. Alternatively, simply set `custom_model` within the
        top level DDPG `Q_model` config key to make this default implementation
        of `build_q_model` use your custom Q-nets.

        Returns:
            TorchModelV2: The TorchModelV2 Q-net sub-model.
        """
        self.concat_obs_and_actions = False

        orig_space = getattr(obs_space, "original_space", obs_space)
        if isinstance(orig_space, Box) and len(orig_space.shape) == 1:
            input_space = Box(
                float("-inf"),
                float("inf"),
                shape=(orig_space.shape[0] + action_space.shape[0], ))
            self.concat_obs_and_actions = True
        else:
            if isinstance(orig_space, gym.spaces.Tuple):
                spaces = list(orig_space.spaces)
            elif isinstance(orig_space, gym.spaces.Dict):
                spaces = list(orig_space.spaces.values())
            else:
                spaces = [obs_space]
            input_space = gym.spaces.Tuple(spaces + [action_space])

        model = ModelCatalog.get_model_v2(
            input_space,
            action_space,
            num_outputs,
            q_model_config,
            framework="torch",
            name=name)
        return model

    @override(DDPGTorchModel)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType):
        """The common (Q-net and policy-net) forward pass.

        NOTE: It is not(!) recommended to override this method as it would
        introduce a shared pre-network, which would be updated by both
        actor- and critic optimizers.

        For rnn support remove input_dict filter and pass state and seq_lens
        """
        model_out = {"obs": input_dict[SampleBatch.OBS]}

        if self.use_prev_action:
            model_out["prev_actions"] = input_dict[SampleBatch.PREV_ACTIONS]
        if self.use_prev_reward:
            model_out["prev_rewards"] = input_dict[SampleBatch.PREV_REWARDS]

        return model_out, state

    def _get_q_value(self, model_out: TensorType, actions, net,
                     state_in: List[TensorType],
                     seq_lens: TensorType):
        # Continuous case -> concat actions to model_out.
        model_out = copy.deepcopy(model_out)
        if actions is not None:
            if self.concat_obs_and_actions:
                model_out[SampleBatch.OBS] = \
                    torch.cat([model_out[SampleBatch.OBS], actions], dim=-1)
            else:
                model_out[SampleBatch.OBS] = \
                    force_list(model_out[SampleBatch.OBS]) + [actions]
        else:
            actions = torch.zeros(
                list(model_out[SampleBatch.OBS].shape[:-1]) + [self.action_dim])
            if self.concat_obs_and_actions:
                model_out[SampleBatch.OBS] = \
                    torch.cat([model_out[SampleBatch.OBS], actions], dim=-1)
            else:
                model_out[SampleBatch.OBS] = \
                    force_list(model_out[SampleBatch.OBS]) + [actions]

        # Switch on training mode (when getting Q-values, we are usually in
        # training).
        model_out["is_training"] = True

        out, state_out = net(model_out, state_in, seq_lens)
        return out, state_out

    @override(DDPGTorchModel)
    def get_q_values(self,
                     model_out: TensorType,
                     state_in: List[TensorType],
                     seq_lens: TensorType,
                     actions: Optional[TensorType] = None) -> TensorType:
        return self._get_q_value(model_out, actions, self.q_model, state_in,
                                 seq_lens)

    @override(DDPGTorchModel)
    def get_twin_q_values(self,
                          model_out: TensorType,
                          state_in: List[TensorType],
                          seq_lens: TensorType,
                          actions: Optional[TensorType] = None) -> TensorType:
        return self._get_q_value(model_out, actions, self.twin_q_model, state_in,
                                 seq_lens)

    @override(DDPGTorchModel)
    def get_policy_output(
            self, model_out: TensorType, state_in: List[TensorType],
            seq_lens: TensorType):
        return self.policy_model(model_out, state_in, seq_lens)

    @override(ModelV2)
    def get_initial_state(self):
        policy_initial_state = self.policy_model.get_initial_state()
        q_initial_state = self.q_model.get_initial_state()
        if self.twin_q_model:
            q_initial_state *= 2
        return policy_initial_state + q_initial_state

    def select_state(self, state_batch: List[TensorType],
                     net: List[str]) -> Dict[str, List[TensorType]]:
        assert all(n in ["policy", "q", "twin_q"] for n in net), \
            "Selected state must be either for policy, q or twin_q network"
        policy_state_len = len(self.policy_model.get_initial_state())
        q_state_len = len(self.q_model.get_initial_state())

        selected_state = {}
        for n in net:
            if n == "policy":
                selected_state[n] = state_batch[:policy_state_len]
            elif n == "q":
                selected_state[n] = state_batch[policy_state_len:
                                                policy_state_len + q_state_len]
            elif n == "twin_q":
                if self.twin_q_model:
                    selected_state[n] = state_batch[policy_state_len +
                                                    q_state_len:]
                else:
                    selected_state[n] = []
        return selected_state
