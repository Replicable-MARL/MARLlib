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

from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_ops import sequence_mask
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from marllib.marl.algos.utils.setup_utils import get_device
from marllib.marl.algos.utils.manipulate_tensor import flat_grad, flat_params, flat_hessian

torch, nn = try_import_torch()


class TrustRegionUpdator:
    kl_threshold = 0.01
    ls_step = 15
    accept_ratio = 0.1
    back_ratio = 0.8
    atol = 1e-7
    critic_lr = 5e-3

    # delta = 0.01

    def __init__(self, model, dist_class, train_batch, adv_targ, initialize_policy_loss, initialize_critic_loss=None):
        self.model = model
        self.dist_class = dist_class
        self.train_batch = train_batch
        self.adv_targ = adv_targ
        self.initialize_policy_loss = initialize_policy_loss
        self.initialize_critic_loss = initialize_critic_loss
        self.stored_actor_parameters = None
        self.device = get_device()

    @property
    def actor_parameters(self):
        return self.model.actor_parameters()

    @property
    def loss(self):
        logits, state = self.model(self.train_batch)
        try:
            curr_action_dist = self.dist_class(logits, self.model)
        except ValueError as e:
            print(e)

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

    def set_actor_params(self, new_flat_params):
        vector_to_parameters(new_flat_params, self.actor_parameters)

    def store_current_actor_params(self):
        self.stored_actor_parameters = self.actor_parameters

    def recovery_actor_params_to_before_linear_search(self):
        stored_actor_parameters = flat_params(self.stored_actor_parameters)
        self.set_actor_params(stored_actor_parameters)

    def reset_actor_params(self):
        initialized_actor_parameters = flat_params(self.model.actor_initialized_parameters)
        self.set_actor_params(initialized_actor_parameters)

    def fisher_vector_product(self, p):
        p.detach()
        kl = self.kl.mean()
        kl_grads = torch.autograd.grad(kl, self.actor_parameters, create_graph=True, allow_unused=True)
        kl_grads = flat_grad(kl_grads)
        kl_grad_p = (kl_grads * p).sum()
        kl_hessian_p = torch.autograd.grad(kl_grad_p, self.actor_parameters, allow_unused=True)
        kl_hessian_p = flat_hessian(kl_hessian_p)
        return kl_hessian_p + 0.1 * p

    def conjugate_gradients(self, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size()).to(device=self.device)
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
        with torch.backends.cudnn.flags(enabled=False):
            self.update_actor(self.initialize_policy_loss)
            if update_critic:
                self.update_critic(self.initialize_critic_loss)

    def update_critic(self, critic_loss):
        critic_loss_grad = torch.autograd.grad(critic_loss, self.critic_parameters, allow_unused=True)

        new_params = (
                parameters_to_vector(self.critic_parameters) - flat_grad(
            critic_loss_grad) * TrustRegionUpdator.critic_lr
        )

        vector_to_parameters(new_params, self.critic_parameters)

        return None

    def update_actor(self, policy_loss):

        loss_grad = torch.autograd.grad(policy_loss, self.actor_parameters, allow_unused=True, retain_graph=True)
        pol_grad = flat_grad(loss_grad)
        step_dir = self.conjugate_gradients(
            b=pol_grad.data,
            nsteps=10,
        )

        fisher_norm = pol_grad.dot(step_dir)
        scala = 0 if fisher_norm < 0 else torch.sqrt(2 * self.kl_threshold / (fisher_norm + 1e-8))
        full_step = scala * step_dir
        loss = policy_loss.data.cpu().numpy()
        params = flat_grad(self.actor_parameters)
        self.store_current_actor_params()

        expected_improve = pol_grad.dot(full_step).item()
        linear_search_updated = False
        fraction = 1
        if expected_improve >= self.atol:
            for i in range(self.ls_step):
                new_params = params + fraction * full_step
                self.set_actor_params(new_params)
                new_loss = self.loss.data.cpu().numpy()
                loss_improve = new_loss - loss
                kl = self.kl.mean()
                if kl < self.kl_threshold and (loss_improve / expected_improve) >= self.accept_ratio and \
                        loss_improve.item() > 0:
                    linear_search_updated = True
                    break
                else:
                    expected_improve *= self.back_ratio
                    fraction *= self.back_ratio

            if not linear_search_updated:
                self.recovery_actor_params_to_before_linear_search()
