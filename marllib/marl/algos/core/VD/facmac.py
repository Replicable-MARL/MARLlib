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

from marllib.marl.algos.core.IL.ddpg import *
from marllib.marl.algos.utils.mixing_Q import q_value_mixing, MixingQValueMixin, before_learn_on_batch
from ray.rllib.agents.ddpg.ddpg_torch_policy import TargetNetworkMixin, ComputeTDErrorMixin

torch, nn = try_import_torch()


def build_facmac_models(policy, observation_space, action_space, config):
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
        default_model=FACMAC_TorchModel,
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
        default_model=FACMAC_TorchModel,
        name="iddpg_model",
        policy_model_config=policy_model_config,
        q_model_config=q_model_config,
        twin_q=config["twin_q"],
        add_layer_norm=(policy.config["exploration_config"].get("type") ==
                        "ParameterNoise"),
    )

    return policy.model


def build_facmac_models_and_action_dist(
        policy: Policy, obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: TrainerConfigDict) -> Tuple[ModelV2, ActionDistribution]:
    model = build_facmac_models(policy, obs_space, action_space, config)

    assert model.get_initial_state() != [], \
        "RNNDDPG requires its model to be a recurrent one!"

    if isinstance(action_space, Simplex):
        return model, TorchDirichlet
    else:
        return model, TorchDeterministic


class FACMAC_TorchModel(IDDPGTorchModel):
    """
    Data flow:
        obs -> forward() -> model_out
        model_out -> get_policy_output() -> pi(s)
        model_out, actions -> get_q_values() -> Q(s, a)
        model_out, actions -> get_twin_q_values() -> Q_twin(s, a)

    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass.
    """

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
        if "opponent_q" in input_dict:  # add additional info to model out
            model_out["opponent_q"] = input_dict["opponent_q"]
            model_out["state"] = input_dict["state"]
        else:
            o = input_dict["obs"]
            if self.state_flag:
                model_out["state"] = torch.zeros_like(o["state"], dtype=o["state"].dtype)
            else:
                model_out["state"] = torch.zeros((o["obs"].shape[0], self.num_agents, o["obs"].shape[1]),
                                                 dtype=o["obs"].dtype)
            model_out["opponent_q"] = torch.zeros((o["obs"].shape[0], self.num_agents - 1),
                                                  dtype=o["obs"].dtype)

        if self.use_prev_action:
            model_out["prev_actions"] = input_dict[SampleBatch.PREV_ACTIONS]
        if self.use_prev_reward:
            model_out["prev_rewards"] = input_dict[SampleBatch.PREV_REWARDS]

        return model_out, state

    def _get_q_value_and_mixing(self, model_out: TensorType,
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
        mixing_out = net.mixing_value(torch.cat((out, model_out["opponent_q"]), 1), model_out["state"])
        return mixing_out, state_out

    def get_q_values_and_mixing(self,
                                model_out: TensorType,
                                state_in: List[TensorType],
                                seq_lens: TensorType,
                                actions: Optional[TensorType] = None) -> TensorType:
        return self._get_q_value_and_mixing(model_out, state_in, self.q_model, actions,
                                            seq_lens)


# Copied from rnnddpg but optimizing the central q function.
def value_mixing_ddpg_loss(policy, model, dist_class, train_batch):
    MixingQValueMixin.__init__(policy)
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
        "state": train_batch["state"],
        "is_training": True,
        "opponent_q": train_batch["opponent_q"],
        "prev_actions": train_batch[SampleBatch.PREV_ACTIONS],
        "prev_rewards": train_batch[SampleBatch.PREV_REWARDS],
    }
    model_out_t, state_in_t = model(input_dict, state_batches, seq_lens)
    states_in_t = model.select_state(state_in_t, ["policy", "q", "twin_q"])

    input_dict_next = {
        "obs": train_batch[SampleBatch.NEXT_OBS],
        "state": train_batch["new_state"],
        "is_training": True,
        "opponent_q": train_batch["next_opponent_q"],  # this from target Q net
        "prev_actions": train_batch[SampleBatch.ACTIONS],
        "prev_rewards": train_batch[SampleBatch.REWARDS],
    }

    # model_out_tp1, state_in_tp1 = model(
    #     input_dict_next, state_batches, seq_lens)
    # states_in_tp1 = model.select_state(state_in_tp1, ["policy", "q", "twin_q"])

    target_model_out_tp1, target_state_in_tp1 = target_model(
        input_dict_next, state_batches, seq_lens)
    target_states_in_tp1 = target_model.select_state(target_state_in_tp1,
                                                     ["policy", "q", "twin_q"])

    # Policy network evaluation.
    # prev_update_ops = set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS))
    policy_t = model.get_policy_output(
        model_out_t, states_in_t["policy"], seq_lens)[0]
    # policy_batchnorm_update_ops = list(
    #    set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS)) - prev_update_ops)

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
    # prev_update_ops = set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS))
    # Q-values for given actions & observations in given current
    q_t = model.get_q_values_and_mixing(
        model_out_t, states_in_t["q"], seq_lens, train_batch[SampleBatch.ACTIONS])[0]

    # Q-values for current policy (no noise) in given current state
    q_t_det_policy = model.get_q_values_and_mixing(
        model_out_t, states_in_t["q"], seq_lens, policy_t)[0]

    actor_loss = -torch.mean(q_t_det_policy)

    # Target q-net(s) evaluation.
    q_tp1 = target_model.get_q_values_and_mixing(
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

    # Compute the error (potentially clipped).
    td_error = q_t_selected - q_t_selected_target
    td_error = td_error * seq_mask
    if use_huber:
        errors = huber_loss(td_error, huber_threshold)
    else:
        errors = 0.5 * torch.pow(td_error, 2.0)

    critic_loss = torch.mean(train_batch[PRIO_WEIGHTS] * errors)

    # Add l2-regularization if required.
    if l2_reg is not None:
        for name, var in model.policy_variables(as_dict=True).items():
            if "bias" not in name:
                actor_loss += (l2_reg * l2_loss(var))
        for name, var in model.q_variables(as_dict=True).items():
            if "bias" not in name:
                critic_loss += (l2_reg * l2_loss(var))

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


FACMACTorchPolicy = IDDPGTorchPolicy.with_updates(
    name="FACMACTorchPolicy",
    postprocess_fn=q_value_mixing,
    make_model_and_action_dist=build_facmac_models_and_action_dist,
    loss_fn=value_mixing_ddpg_loss,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        MixingQValueMixin
    ]
)


def get_policy_class(config: TrainerConfigDict) -> Optional[Type[Policy]]:
    if config["framework"] == "torch":
        return FACMACTorchPolicy


def validate_config(config: TrainerConfigDict) -> None:
    # Add the `burn_in` to the Model's max_seq_len.
    # Set the replay sequence length to the max_seq_len of the model.
    config["replay_sequence_length"] = \
        config["burn_in"] + config["model"]["max_seq_len"]

    def f(batch, workers, config):
        policies = dict(workers.local_worker()
                        .foreach_trainable_policy(lambda p, i: (i, p)))
        return before_learn_on_batch(batch, policies,
                                     config["train_batch_size"])

    config["before_learn_on_batch"] = f


FACMACTrainer = IDDPGTrainer.with_updates(
    name="FACMACTrainer",
    default_config=IDDPG_DEFAULT_CONFIG,
    default_policy=FACMACTorchPolicy,
    get_policy_class=get_policy_class,
    validate_config=validate_config,
    allow_unknown_subkeys=["Q_model", "policy_model"]
)
