from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()


def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        if grad is None:
            continue
        grad_flatten.append(grad.reshape(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten


def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        if hessian is None:
            continue
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def flat_params(parameters):
    params = []
    for param in parameters:
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten
