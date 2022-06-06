from ray.rllib.utils.framework import try_import_tf, try_import_torch, get_variable
import numpy as np
from functools import partial
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_ops import explained_variance, sequence_mask
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from marl.algos.utils.get_hetero_info import (
    get_global_name,
    contain_global_obs,
    state_name,
    global_state_name,
    TRAINING,
)
from ray.rllib.evaluation.postprocessing import discount_cumsum, Postprocessing
from icecream import ic

torch, nn = try_import_torch()


def update_model(self, model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length


class HATRPOUpdator:
    def __init__(self, model, models, dist_class, train_batch, mean_fn, adv_targ, initialize_policy_loss, initialize_critic_loss):
        self.updaters = []

        assert model == models[-1]

        main_model = model

        self.main_model = main_model

        self.dist_class = dist_class

        self.mean_fn = mean_fn

        m_advantage = train_batch[Postprocessing.ADVANTAGES]

        random_indices = np.random.permutation(len(models))

        for i in random_indices:
            current_model = models[i]
            if current_model is main_model:
                logits, state = current_model(train_batch)
                current_action_dist = dist_class(logits, current_model)
                old_action_log_dist = train_batch[SampleBatch.ACTION_LOGP]
                actions = train_batch[SampleBatch.ACTIONS]
                importance_sampling = torch.exp(current_action_dist.logp(actions) - old_action_log_dist)
                agent_train_batch = train_batch
            else:
                agent_train_batch, importance_sampling = self.get_sub_train_batch(train_batch, i)

            policy_loss = self.mean_fn(m_advantage * importance_sampling)

            self.updaters.append(
                TrustRegionUpdator(
                    current_model, dist_class, agent_train_batch, m_advantage, policy_loss, initialize_critic_loss
                )
            )

            m_advantage = importance_sampling * m_advantage

    def update(self):
        for updater in self.updaters:
            if updater.model is self.main_model:
                updater.update(update_critic=True)
            else:
                updater.update(update_critic=False)

    def get_sub_train_batch(self, train_batch, agent_id):
        current_action_logits = train_batch[
            get_global_name(SampleBatch.ACTION_DIST_INPUTS, agent_id)
        ]

        current_action_dist = self.dist_class(current_action_logits, None)

        old_action_log_dist = train_batch[
            get_global_name(SampleBatch.ACTION_LOGP, agent_id)
        ]

        actions = train_batch[get_global_name(SampleBatch.ACTIONS, agent_id)]

        obs = train_batch[get_global_name(SampleBatch.OBS, agent_id)]

        train_batch_for_trpo_update = SampleBatch(
            obs=obs,
            seq_lens=train_batch[SampleBatch.SEQ_LENS]
        )

        train_batch_for_trpo_update.is_training = bool(train_batch[get_global_name(TRAINING, agent_id)][0])

        i = 0

        while state_name(i) in train_batch:
            agent_state_name = global_state_name(i, agent_id)
            train_batch_for_trpo_update[state_name(i)] = train_batch[agent_state_name]
            i += 1

        importance_sampling = torch.exp(current_action_dist.logp(actions) - old_action_log_dist)

        return train_batch_for_trpo_update, importance_sampling

    def is_main_model(self, m):
        return m is self.main_model


class TrustRegionUpdator:

    kl_threshold = 0.01
    ls_step = 15
    accept_ratio = 0.1
    back_ratio = 0.8
    atol = 1e-7
    # delta = 0.01

    def __init__(self, model, dist_class, train_batch, adv_targ, initialize_policy_loss, initialize_critic_loss):
        self.model = model
        self.dist_class = dist_class
        self.train_batch = train_batch
        self.adv_targ = adv_targ
        self.initialize_policy_loss = initialize_policy_loss
        self.initialize_critic_loss = initialize_critic_loss
        self.stored_actor_parameters = None

    @property
    def actor_parameters(self):
        return self.model.actor_parameters()

    @property
    def loss(self):
        logits, state = self.model(self.train_batch)
        curr_action_dist = self.dist_class(logits, self.model)

        logp_ratio = torch.exp(
            curr_action_dist.logp(self.train_batch[SampleBatch.ACTIONS]) -
            self.train_batch[SampleBatch.ACTION_LOGP]
        )

        if state:
            B = len(self.train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                self.train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=self.model.is_time_major()
            )
            mask = torch.reshape(mask, [-1])

            loss = (torch.sum(logp_ratio * self.adv_targ, dim=-1, keepdim=True) *
                    mask).sum() / mask.sum()
        else:
            loss = torch.sum(logp_ratio * self.adv_targ, dim=-1, keepdim=True).mean()

        new_loss = loss

        return new_loss

    @property
    def kl(self):
        _logits, _state = self.model(self.train_batch)
        _curr_action_dist = self.dist_class(_logits, self.model)
        action_dist_inputs = self.train_batch[SampleBatch.ACTION_DIST_INPUTS]
        _prev_action_dist = self.dist_class(action_dist_inputs, self.model)

        kl = _prev_action_dist.kl(_curr_action_dist)

        return kl

    @property
    def entropy(self):
        _logits, _state = self.model(self.train_batch)
        _curr_action_dist = self.dist_class(_logits, self.model)
        curr_entropy = _curr_action_dist.entropy()
        return curr_entropy

    @property
    def critic_parameters(self):
        return self.model.critic_parameters()

    @staticmethod
    def flat_grad(grads):
        grad_flatten = []
        for grad in grads:
            if grad is None:
                continue
            grad_flatten.append(grad.view(-1))
        grad_flatten = torch.cat(grad_flatten)
        return grad_flatten

    @staticmethod
    def flat_hessian(hessians):
        hessians_flatten = []
        for hessian in hessians:
            if hessian is None:
                continue
            hessians_flatten.append(hessian.contiguous().view(-1))
        hessians_flatten = torch.cat(hessians_flatten).data
        return hessians_flatten

    @staticmethod
    def flat_params(parameters):
        params = []
        for param in parameters:
            params.append(param.data.view(-1))
        params_flatten = torch.cat(params)
        return params_flatten

    def set_actor_params(self, new_flat_params):
        vector_to_parameters(new_flat_params, self.actor_parameters)
        # prev_ind = 0
        # index = 0
        # for params in self.actor_parameters:
        #     params_length = len(params.view(-1))
        #     new_param = new_flat_params[index: index + params_length]
        #     if torch.any(new_param.isnan()):
        #         print('find nan parameters')
        #     new_param = new_param.view(params.size())
        #     params.data.copy_(new_param)
        #     index += params_length

    def store_current_actor_params(self):
        self.stored_actor_parameters = self.actor_parameters

    def recovery_actor_params_to_before_linear_search(self):
        stored_actor_parameters = self.flat_params(self.stored_actor_parameters)
        self.set_actor_params(stored_actor_parameters)

    def reset_actor_params(self):
        initialized_actor_parameters = self.flat_params(self.model.actor_initialized_parameters)
        self.set_actor_params(initialized_actor_parameters)

    def fisher_vector_product(self, p):
        p.detach()
        kl = self.kl.mean()
        # en = self.entropy.mean()

        kl_grads = torch.autograd.grad(kl, self.actor_parameters, create_graph=True, allow_unused=True)
        kl_grads = self.flat_grad(kl_grads)

        kl_grad_p = (kl_grads * p).sum()
        # kl_hessian_p = torch.autograd.grad(kl_grad_p, self.actor_parameters, allow_unused=True, retain_graph=True)
        kl_hessian_p = torch.autograd.grad(kl_grad_p, self.actor_parameters, allow_unused=True)
        kl_hessian_p = self.flat_hessian(kl_hessian_p)

        return kl_hessian_p + 0.1 * p

    def conjugate_gradients(self, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = self.fisher_vector_product(p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def update(self, update_critic=True):
        self.update_actor(self.initialize_policy_loss)
        if update_critic:
            self.update_critic(self.initialize_critic_loss)

    def update_critic(self, critic_loss):
        critic_loss_grad = torch.autograd.grad(critic_loss, self.critic_parameters, allow_unused=True)

        lr = 5e-3

        new_params = (
            parameters_to_vector(self.critic_parameters) - self.flat_grad(critic_loss_grad) * lr
        )

        vector_to_parameters(new_params, self.critic_parameters)

        return None

    def update_actor(self, policy_loss):

        loss_grad = torch.autograd.grad(policy_loss, self.actor_parameters, allow_unused=True)
        pol_grad = self.flat_grad(loss_grad)

        # assert not torch.all(pol_grad) == 0

        step_dir = self.conjugate_gradients(
            b=pol_grad.data,
            nsteps=10,
        )

        fisher_norm = pol_grad.dot(step_dir)

        scala = 0 if fisher_norm < 0 else torch.sqrt(2 * self.kl_threshold / (fisher_norm + 1e-8))

        full_step = scala * step_dir

        loss = policy_loss.data.cpu().numpy()
        params = self.flat_grad(self.actor_parameters)

        # fvp = self.fisher_vector_product(p=step_dir)

        # shs = 0.5 * (step_dir * fvp).sum(0, keepdim=True)

        # assert shs > 0, f'shs = {shs}'

        # step_size = 1 / torch.sqrt(shs / self.kl_threshold)[0]
        # full_step = step_size * step_dir

        # self.reset_actor_params()
        self.store_current_actor_params()

        expected_improve = pol_grad.dot(full_step).item()
        # expected_improve = (loss_grad * full_step).sum(0, keepdim=True)
        # expected_improve = expected_improve.data.cpu().numpy()
        # ic(expected_improve)

        linear_search_updated = False
        fraction = 1

        # print()
        if expected_improve >= self.atol:
            # print('linear search:')
            for i in range(self.ls_step):
                # print(f'\t{i}/{TrustRegionUpdator.ls_step}', end='')
                new_params = params + fraction * full_step
                self.set_actor_params(new_params)

                new_loss = self.loss.data.cpu().numpy()

                loss_improve = new_loss - loss

                kl = self.kl.mean()

                # ic(kl)
                # ic(loss_improve / expected_improve)
                # ic(loss_improve.item())

                if kl < self.kl_threshold and (loss_improve / expected_improve) >= self.accept_ratio and \
                        loss_improve.item() > 0:
                    linear_search_updated = True
                    break
                else:
                    expected_improve *= self.back_ratio
                    fraction *= self.back_ratio

            if not linear_search_updated:
                self.recovery_actor_params_to_before_linear_search()
            # if not finish:
            #     self.reset_actor_params()

