from ray.rllib.utils.framework import try_import_tf, try_import_torch, get_variable
import numpy as np
from functools import partial

torch, nn = try_import_torch()


def Fvp(v, model, kl_f):

    v.detach()

    # grads = torch.autograd.grad(kl_mean, model.parameters(), create_graph=True, retain_graph=True)
    kl = kl_f(mean=True)
    grads = torch.autograd.grad(kl, model.actor_parameters(), create_graph=True, allow_unused=True)
    flat_grad_kl = torch.cat([g.view(-1) for g in grads if g is not None])

    kl_v = (flat_grad_kl * v).sum()
    grads = torch.autograd.grad(kl_v, model.actor_parameters(), allow_unused=True, retain_graph=True)
    # grads = torch.autograd.grad(kl_v, model.parameters())
    flat_grad_grad_kl = torch.cat([g.contiguous().view(-1) for g in grads if g is not None]).data

    damping = 0.1
    min_step = 1e-2

    def get_res(d):
        return flat_grad_grad_kl + v * d

    # res = flat_grad_grad_kl + v * damping
    res = get_res(damping)

    while (v * res).sum(0, keepdim=True) < 0:
        damping += min_step
        res = get_res(damping)

    return res


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
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
               loss_f,
               kl_f,
               x,
               fullstep,
               expected_improve_rate,
               max_backtracks=10,
               accept_ratio=.1,
               kl_threshold=None
               ):
    fval = loss_f().data # notice, need a a minus to loss function.
    # print('fval === ', fval)
    # print("fval before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        # stepfrac: 1, 0.5, 0.5 ** 2, 0.5 ** 3, ..
        xnew = x + stepfrac * fullstep
        set_flat_actor_params_to(model, xnew)
        newfval = loss_f().data
        # print('new-fval === ', newfval)
        # actual_improve = fval - newfval
        actual_improve = newfval - fval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        # print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

        kl = kl_f(mean=True)

        if kl < kl_threshold and ratio.item() > accept_ratio and actual_improve.item() > 0:
            # print("fval after", newfval.item())
            return True, xnew

    return False, x


def update_model_use_trust_region(model, train_batch, advantages,
                                  actions, action_logp, action_dist_inputs, dist_class, mean_fn):
    # reference: https://github.com/ikostrikov/pytorch-trpo

    # obs = train_batch['obs']
    # if isinstance(train_batch['obs'], dict):
    #     obs = obs['obs']
    # print(f'train OBS.shape is {obs.shape}')

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

        _policy_loss = advantages * _logp_ratio

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

    stepdir = conjugate_gradients(c_Fvp, loss_grad, 10)

    shs = 0.5 * (stepdir * c_Fvp(stepdir)).sum(0, keepdim=True)

    max_kl = 1e-2
    step_size = 1 / torch.sqrt(shs / max_kl)[0]
    full_step = step_size * stepdir

    # neg_dot_stepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
    excepted_improve = (loss_grad * stepdir).sum(0, keepdim=True)
    excepted_improve = excepted_improve.data.cpu().numpy()

    # prev_params = torch.cat([p.data.view(-1) for p in model.parameters()])
    prev_params = torch.cat([p.data.view(-1) for p in model.actor_parameters()])

    _get_mean_and_no_grad_policy_loss = partial(_get_policy_loss, volatile=True, mean=True)
    success, new_params = linesearch(model, loss_f=_get_mean_and_no_grad_policy_loss,
                                     kl_f=_get_kl, x=prev_params, fullstep=full_step,
                                     expected_improve_rate=excepted_improve, kl_threshold=max_kl
                                     )

    if success:
        set_flat_actor_params_to(model, new_params)

    return action_kl, policy_loss_for_rllib
