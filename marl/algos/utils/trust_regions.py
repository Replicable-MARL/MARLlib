from ray.rllib.utils.framework import try_import_tf, try_import_torch, get_variable
import numpy as np
from functools import partial

torch, nn = try_import_torch()


def Fvp(v, model, kl_f):

    # grads = torch.autograd.grad(kl_mean, model.parameters(), create_graph=True, retain_graph=True)
    kl = kl_f(mean=True)
    grads = torch.autograd.grad(kl, model.parameters(), create_graph=True, allow_unused=True)
    flat_grad_kl = torch.cat([g.view(-1) for g in grads if g is not None])

    kl_v = (flat_grad_kl * v).sum()
    grads = torch.autograd.grad(kl_v, model.parameters(), allow_unused=True, retain_graph=True)
    # grads = torch.autograd.grad(kl_v, model.parameters())
    flat_grad_grad_kl = torch.cat([g.contiguous().view(-1) for g in grads if g is not None]).data

    damping = 0.1
    return flat_grad_grad_kl + v * damping


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        if rdotr.isnan() or any(_Avp.isnan()) or any(p.isnan()):
            print('Find Nan')
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


def set_flat_actor_params_to(model, flat_params):
    prev_ind = 0
    for param in model.actor_parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


@torch.no_grad()
def linesearch(model,
               f,
               x,
               fullstep,
               expected_improve_rate,
               max_backtracks=10,
               accept_ratio=.1):
    fval = f().data # notice, need a a minus to loss function.
    # print('fval === ', fval)
    print("fval before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        if any(torch.isnan(xnew)):
            print(xnew)
        set_flat_actor_params_to(model, xnew)
        newfval = f().data
        print('new-fval === ', newfval)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        # print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            print("fval after", newfval.item())
            return True, xnew
    return False, x


def update_model_use_trust_region(model, train_batch, advantages, obs, actions, action_logp, action_dist_inputs, dist_class, mean_fn):
    # reference: https://github.com/ikostrikov/pytorch-trpo

    __obs = {'obs': {'obs': obs}} # to match network forward

    def _get_policy_loss(volatile=False, mean=False):
        if volatile:
            with torch.no_grad():
                _logits, _state = model(train_batch)
        else:
            _logits, _state = model(train_batch)

        _curr_action_dist = dist_class(_logits, model)
        _logp_ratio = torch.exp(
            _curr_action_dist.logp(actions) -
            action_logp,
        )

        _policy_loss = -advantages * _logp_ratio

        return mean_fn(_policy_loss) if mean else _policy_loss

    def _get_kl(mean=False):
        _logits, _state = model(train_batch)
        _curr_action_dist = dist_class(_logits, model)
        _prev_action_dist = dist_class(action_dist_inputs, model)
        kl = _prev_action_dist.kl(_curr_action_dist)
        return mean_fn(kl) if mean else kl

    # policy_loss = _get_policy_loss()

    action_kl = _get_kl()
    policy_loss_for_rllib = _get_policy_loss(mean=False)

    grads = torch.autograd.grad(mean_fn(policy_loss_for_rllib), model.parameters(),
                                allow_unused=True,
                                retain_graph=True,
    )

    loss_grad = torch.cat([grad.view(-1) for grad in grads if grad is not None]).data

    c_Fvp = partial(Fvp, model=model, kl_f=_get_kl)
    # current model fvp

    stepdir = conjugate_gradients(c_Fvp, -loss_grad, 10)

    shs = 0.5 * (stepdir * c_Fvp(stepdir)).sum(0, keepdim=True)

    max_kl = 1e-2

    lm = torch.sqrt(shs / max_kl)

    fullstep = stepdir / lm[0]

    neg_dot_stepdir = (-loss_grad * stepdir).sum(0, keepdim=True)

    # prev_params = torch.cat([p.data.view(-1) for p in model.parameters()])
    prev_params = torch.cat([p.data.view(-1) for p in model.actor_parameters()])

    _get_mean_and_no_grad_policy_loss = partial(_get_policy_loss, volatile=True, mean=True)
    success, new_params = linesearch(model, _get_mean_and_no_grad_policy_loss,
                                     prev_params, fullstep, neg_dot_stepdir / lm[0]
                                     )

    if success:
        set_flat_actor_params_to(model, new_params)

    return action_kl, policy_loss_for_rllib
