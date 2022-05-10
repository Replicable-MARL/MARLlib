import torch
from torch.autograd import grad
from typing import Dict, Iterator, Union
from torch import Tensor
import functools
from typing import Any, Callable
import numpy as np

LEARNER_STATS_KEY = "learner_stats"


def _flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        if grad is None:
            continue
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten


def _flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        if hessian is None:
            continue
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def _flat_params(params):
    _params = []
    for param in params:
        _params.append(param.data.view(-1))

    params_flatten = torch.cat(_params)
    return params_flatten


def _fisher_vector_product(policy, params, p):
    p.detach()

    logits, state = policy.model(policy.train_batch)
    curr_action_dist = policy.dist_class(logits, policy.model)

    kl_mean = policy.reduce_mean(policy.prev_action_dist.kl(curr_action_dist))
    kl_grad = torch.autograd.grad(kl_mean, params, create_graph=True, allow_unused=True)
    kl_grad = _flat_grad(kl_grad)
    kl_grad_p = (kl_grad * p).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, params, allow_unused=True)
    kl_hessian_p = _flat_hessian(kl_hessian_p)

    return kl_hessian_p + 0.1 * p


def _conjugate_gradient(policy, params, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = _fisher_vector_product(policy, params, p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
        # print(f'Conjugate : {i} success!')
    return x


@torch.no_grad()
def line_search(
        func,
        x_0,
        d_x,
        expected_improvement,
        y_0=None,
        accept_ratio=0.1,
        backtrack_ratio=0.8,
        max_backtracks=15,
        atol=1e-7,
):
    """Perform a linesearch on func with start x_0 and direction d_x."""
    # pylint:disable=too-many-arguments
    if y_0 is None:
        y_0 = func(x_0)

    if expected_improvement >= atol:
        for exp in range(max_backtracks):
            ratio = backtrack_ratio ** exp
            x_new = x_0 - ratio * d_x
            y_new = func(x_new)
            improvement = y_0 - y_new
            # Armijo condition
            if improvement / (expected_improvement * ratio) >= accept_ratio:
                return x_new, expected_improvement * ratio, improvement

    return x_0, expected_improvement, 0


def learner_stats(func: Callable[[Any], dict]) -> Callable[[Any], dict]:
    """Wrap function to return stats under learner stats key."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        stats = func(*args, **kwargs)
        nested = stats.get(LEARNER_STATS_KEY, {})
        unnested = {k: v for k, v in stats.items() if k != LEARNER_STATS_KEY}
        return {LEARNER_STATS_KEY: {**nested, **unnested}}

    return wrapped


def get_keys(mapping, *keys):
    """Return the values corresponding to the given keys, in order."""
    return (mapping[k] for k in keys)


def explained_variance(targets, pred):
    """Compute the explained variance given targets and predictions."""
    targets_var = np.var(targets, axis=0)
    diff_var = np.var(targets - pred, axis=0)
    return np.maximum(-1.0, 1.0 - (diff_var / targets_var))
