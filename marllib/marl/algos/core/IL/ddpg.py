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

from typing import Type
from ray.rllib.agents.ddpg.ddpg import DDPGTrainer, DEFAULT_CONFIG as DDPG_DEFAULT_CONFIG
from ray.rllib.agents.ddpg.ddpg_torch_policy import DDPGTorchPolicy
import copy
import gym
from typing import Tuple
from ray.rllib.agents.dqn.dqn_tf_policy import PRIO_WEIGHTS
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_action_dist import TorchDeterministic, TorchDirichlet
from ray.rllib.models.tf.tf_action_dist import Deterministic
from ray.rllib.policy.policy import Policy
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.torch_ops import huber_loss, l2_loss, sequence_mask
from ray.rllib.utils.typing import TrainerConfigDict, ModelInputDict
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.agents.ddpg.ddpg_torch_model import DDPGTorchModel
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.execution.replay_buffer import *

from marllib.marl.algos.utils.episode_execution_plan import episode_execution_plan

torch, nn = try_import_torch()


def ddpg_actor_critic_loss(policy: Policy, model: ModelV2, _,
                           train_batch: SampleBatch) -> TensorType:
    """Constructs the loss for DDPG Objective.
    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    target_model = policy.target_models[model]

    i = 0
    state_batches = []
    while "state_in_{}".format(i) in train_batch:
        state_batches.append(train_batch["state_in_{}".format(i)])
        i += 1
    assert state_batches
    seq_lens = train_batch.get(SampleBatch.SEQ_LENS)

    twin_q = policy.config["twin_q"]
    gamma = policy.config["gamma"]
    n_step = policy.config["n_step"]
    use_huber = policy.config["use_huber"]
    huber_threshold = policy.config["huber_threshold"]
    l2_reg = policy.config["l2_reg"]

    input_dict = {
        "obs": train_batch[SampleBatch.CUR_OBS],
        "is_training": True,
        "prev_actions": train_batch[SampleBatch.PREV_ACTIONS],
        "prev_rewards": train_batch[SampleBatch.PREV_REWARDS],
    }
    model_out_t, state_in_t = model(input_dict, state_batches, seq_lens)
    states_in_t = model.select_state(state_in_t, ["policy", "q", "twin_q"])

    input_dict_next = {
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "is_training": True,
        "prev_actions": train_batch[SampleBatch.ACTIONS],
        "prev_rewards": train_batch[SampleBatch.REWARDS],
    }

    target_model_out_tp1, target_state_in_tp1 = target_model(
        input_dict_next, state_batches, seq_lens)
    target_states_in_tp1 = target_model.select_state(target_state_in_tp1,
                                                     ["policy", "q", "twin_q"])

    # Policy network evaluation.
    policy_t = model.get_policy_output(
        model_out_t, states_in_t["policy"], seq_lens)[0]

    policy_tp1 = target_model.get_policy_output(
        target_model_out_tp1, target_states_in_tp1["policy"], seq_lens)[0]

    # Action outputs.
    if policy.config["smooth_target_policy"]:
        target_noise_clip = policy.config["target_noise_clip"]
        clipped_normal_sample = torch.clamp(
            torch.normal(
                mean=torch.zeros(policy_tp1.size()),
                std=policy.config["target_noise"]).to(policy_tp1.device),
            -target_noise_clip, target_noise_clip)

        policy_tp1_smoothed = torch.min(
            torch.max(
                policy_tp1 + clipped_normal_sample,
                torch.tensor(
                    policy.action_space.low,
                    dtype=torch.float32,
                    device=policy_tp1.device)),
            torch.tensor(
                policy.action_space.high,
                dtype=torch.float32,
                device=policy_tp1.device))
    else:
        # No smoothing, just use deterministic actions.
        policy_tp1_smoothed = policy_tp1

    # Q-net(s) evaluation.
    # Q-values for given actions & observations in given current
    q_t = model.get_q_values(
        model_out_t, states_in_t["q"], seq_lens, train_batch[SampleBatch.ACTIONS])[0]

    # Q-values for current policy (no noise) in given current state
    q_t_det_policy = model.get_q_values(
        model_out_t, states_in_t["q"], seq_lens, policy_t)[0]
    q_t_det_policy = torch.squeeze(input=q_t_det_policy, axis=len(q_t_det_policy.shape) - 1)

    # Target q-net(s) evaluation.
    q_tp1 = target_model.get_q_values(
        target_model_out_tp1, target_states_in_tp1["q"], seq_lens, policy_tp1_smoothed)[0]

    q_t_selected = torch.squeeze(q_t, axis=len(q_t.shape) - 1)

    q_tp1_best = torch.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
    q_tp1_best_masked = \
        (1.0 - train_batch[SampleBatch.DONES].float()) * \
        q_tp1_best

    # Compute RHS of bellman equation.
    q_t_selected_target = (train_batch[SampleBatch.REWARDS] +
                           gamma ** n_step * q_tp1_best_masked).detach()

    # BURNIN #
    B = state_batches[0].shape[0]
    T = q_t_selected.shape[0] // B
    seq_mask = sequence_mask(train_batch[SampleBatch.SEQ_LENS], T)
    # Mask away also the burn-in sequence at the beginning.
    burn_in = policy.config["burn_in"]
    if burn_in > 0 and burn_in < T:
        seq_mask[:, :burn_in] = False

    seq_mask = seq_mask.reshape(-1)
    num_valid = torch.sum(seq_mask)

    def reduce_mean_valid(t):
        return torch.sum(t[seq_mask]) / num_valid

    # Compute the error (potentially clipped).
    td_error = q_t_selected - q_t_selected_target
    td_error = td_error * seq_mask
    if use_huber:
        errors = huber_loss(td_error, huber_threshold)
    else:
        errors = 0.5 * torch.pow(td_error, 2.0)

    critic_loss = torch.mean(train_batch[PRIO_WEIGHTS] * errors)
    actor_loss = -torch.mean(q_t_det_policy * seq_mask)

    # Add l2-regularization if required.
    if l2_reg is not None:
        for name, var in model.policy_variables(as_dict=True).items():
            if "bias" not in name:
                actor_loss += (l2_reg * l2_loss(var))
        for name, var in model.q_variables(as_dict=True).items():
            if "bias" not in name:
                critic_loss += (l2_reg * l2_loss(var))

    # Model self-supervised losses.
    if policy.config["use_state_preprocessor"]:
        # Expand input_dict in case custom_loss' need them.
        input_dict[SampleBatch.ACTIONS] = train_batch[SampleBatch.ACTIONS]
        input_dict[SampleBatch.REWARDS] = train_batch[SampleBatch.REWARDS]
        input_dict[SampleBatch.DONES] = train_batch[SampleBatch.DONES]
        input_dict[SampleBatch.NEXT_OBS] = train_batch[SampleBatch.NEXT_OBS]
        [actor_loss, critic_loss] = model.custom_loss(
            [actor_loss, critic_loss], input_dict)

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["q_t"] = q_t * seq_mask[..., None]
    model.tower_stats["actor_loss"] = actor_loss
    model.tower_stats["critic_loss"] = critic_loss
    # TD-error tensor in final stats
    # will be concatenated and retrieved for each individual batch item.
    model.tower_stats["td_error"] = td_error

    # Return two loss terms (corresponding to the two optimizers, we create).
    return actor_loss, critic_loss


def build_iddpg_models(policy, observation_space, action_space, config):
    num_outputs = int(np.product(observation_space.shape))

    policy_model_config = MODEL_DEFAULTS.copy()
    policy_model_config.update(config["policy_model"])
    q_model_config = MODEL_DEFAULTS.copy()
    q_model_config.update(config["Q_model"])

    policy.model = ModelCatalog.get_model_v2(
        obs_space=observation_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        default_model=IDDPGTorchModel,
        name="iddpg_model",
        policy_model_config=policy_model_config,
        q_model_config=q_model_config,
        twin_q=config["twin_q"],
        add_layer_norm=(policy.config["exploration_config"].get("type") ==
                        "ParameterNoise"),
    )

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=observation_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework=config["framework"],
        default_model=IDDPGTorchModel,
        name="iddpg_model",
        policy_model_config=policy_model_config,
        q_model_config=q_model_config,
        twin_q=config["twin_q"],
        add_layer_norm=(policy.config["exploration_config"].get("type") ==
                        "ParameterNoise"),
    )

    return policy.model


def build_iddpg_models_and_action_dist(
        policy: Policy, obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: TrainerConfigDict) -> Tuple[ModelV2, ActionDistribution]:
    model = build_iddpg_models(policy, obs_space, action_space, config)

    assert model.get_initial_state() != [], \
        "IDDPG requires its model to be a recurrent one!"

    if isinstance(action_space, Simplex):
        return model, TorchDirichlet
    else:
        return model, TorchDeterministic


def action_distribution_fn(policy: Policy,
                           model: ModelV2,
                           input_dict: ModelInputDict,
                           *,
                           explore=True,
                           is_training=False,
                           state_batches=None,
                           seq_lens=None,
                           prev_action_batch=None,
                           prev_reward_batch=None,
                           timestep=None,
                           **kwargs):
    # modify input output change
    model_out, state_in = model(input_dict, state_batches, seq_lens)

    states_in = model.select_state(state_in, ["policy", "q", "twin_q"])

    distribution_inputs, policy_state_out = \
        model.get_policy_output(model_out, states_in["policy"], seq_lens)
    _, q_state_out = model.get_q_values(model_out, states_in["q"], seq_lens)

    twin_q_state_out = []

    states_out = policy_state_out + q_state_out + twin_q_state_out

    return distribution_inputs, (TorchDeterministic
                                 if policy.config["framework"] == "torch" else
                                 Deterministic), states_out


class IDDPGTorchModel(DDPGTorchModel):
    """Extension of standard DDPGTorchModel for IDDPG.

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

        self.cc_flag = False
        if model_config["custom_model_config"]["algorithm"] in ["maddpg"]:
            self.cc_flag = True
            self.state_flag = model_config["custom_model_config"]["global_state_flag"]

        self.use_prev_action = model_config["lstm_use_prev_action"]

        self.use_prev_reward = model_config["lstm_use_prev_reward"]

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
            self.obs_space, self.action_space, action_outs, {}, "policy_model")

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
                                          q_outs, {}, "q")

        self.twin_q_model = None

        self.state_flag = model_config["custom_model_config"]["global_state_flag"]
        self.num_agents = model_config["custom_model_config"]["num_agents"]

    def build_policy_model(self, obs_space, action_space, num_outputs, policy_model_config,
                           name):
        """Builds the policy model used by this DDPG.

        Override this method in a sub-class of DDPGTFModel to implement your
        own policy net. Alternatively, simply set `custom_model` within the
        top level DDPG `policy_model` config key to make this default
        implementation of `build_policy_model` use your custom policy network.

        Returns:
            TorchModelV2: The TorchModelV2 policy sub-model.
        """
        policy_model_config["custom_model_config"] = self.model_config["custom_model_config"]
        policy_model_config["custom_model"] = "DDPG_Model"
        policy_model_config["custom_model_config"]["extra_action"] = action_space.shape[0]

        model = ModelCatalog.get_model_v2(
            obs_space,
            self.action_space,
            num_outputs,
            policy_model_config,
            framework="torch",
            name=name)
        return model

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

        q_model_config["custom_model_config"] = self.model_config["custom_model_config"]
        q_model_config["custom_model_config"]["extra_action"] = action_space.shape[0]
        q_model_config["custom_model"] = "DDPG_Model"

        model = ModelCatalog.get_model_v2(
            obs_space,
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

    def _get_q_value(self, model_out: TensorType,
                     state_in: List[TensorType],
                     net,
                     actions,
                     seq_lens: TensorType):
        # Continuous case -> concat actions to model_out.
        model_out = copy.deepcopy(model_out)
        if actions is not None:
            model_out["actions"] = actions
        else:
            actions = torch.zeros(
                list(model_out[SampleBatch.OBS]["obs"].shape[:-1]) + [self.action_dim])
            model_out["actions"] = actions.to(state_in[0].device)

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
        return self._get_q_value(model_out, state_in, self.q_model, actions,
                                 seq_lens)

    @override(DDPGTorchModel)
    def get_policy_output(
            self, model_out: TensorType, state_in: List[TensorType],
            seq_lens: TensorType):
        model_out, state_out = self.policy_model(model_out, state_in, seq_lens)
        model_out = self.policy_model.action_out_squashed(model_out)
        return model_out, state_out

    @override(ModelV2)
    def get_initial_state(self):
        policy_initial_state = self.policy_model.get_initial_state()
        q_initial_state = self.q_model.get_initial_state()
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
        return selected_state


logger = logging.getLogger(__name__)

IDDPG_DEFAULT_CONFIG = DDPGTrainer.merge_trainer_configs(
    DDPG_DEFAULT_CONFIG,
    {
        "batch_mode": "complete_episodes",
        "zero_init_states": True,
        "burn_in": 0,
        "replay_sequence_length": -1,
        "Q_model": {},
        "policy_model": {},
        "normalize_actions": False,
        "clip_actions": False,
    },
    _allow_unknown_configs=True,
)


def validate_config(config: TrainerConfigDict) -> None:
    # Add the `burn_in` to the Model's max_seq_len.
    # Set the replay sequence length to the max_seq_len of the model.
    config["replay_sequence_length"] = \
        config["burn_in"] + config["model"]["max_seq_len"]


IDDPGTorchPolicy = DDPGTorchPolicy.with_updates(
    name="IDDPGTorchPolicy",
    get_default_config=lambda: IDDPG_DEFAULT_CONFIG,
    action_distribution_fn=action_distribution_fn,
    make_model_and_action_dist=build_iddpg_models_and_action_dist,
    loss_fn=ddpg_actor_critic_loss,
)


def get_policy_class(config: TrainerConfigDict) -> Optional[Type[Policy]]:
    if config["framework"] == "torch":
        return IDDPGTorchPolicy


IDDPGTrainer = DDPGTrainer.with_updates(
    name="IDDPGTrainer",
    default_config=IDDPG_DEFAULT_CONFIG,
    default_policy=None,
    get_policy_class=get_policy_class,
    validate_config=validate_config,
    allow_unknown_subkeys=["Q_model", "policy_model"],
    execution_plan=episode_execution_plan
)
