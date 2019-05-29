from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def _flatten_convert_none_to_zeros(sequence, like_sequence):
    flat = [
        p.contiguous().view(-1) if p is not None else torch.zeros_like(q).view(-1)
        for p, q in zip(sequence, like_sequence)
    ]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def setup_context(func, context):
    if context is not None:
        if isinstance(context, dict):
            func.set_context(**context)
        elif isinstance(context, list) or isinstance(context, tuple):
            func.set_context(*context)
        else:
            func.set_context(context)


class CondOdeintAdjointMethod(torch.autograd.Function):


    @staticmethod
    def forward(ctx, *args):
        assert len(args) >= 9, 'Internal error: all arguments required.'
        y0, context, ctx_params, func, t, flat_params, rtol, atol, fwd_method, backwd_method, options = \
            args[:-10], args[-10], args[-9], args[-8], args[-7], args[-6], args[-5], args[-4], args[-3], args[-2], args[-1]
        ctx.ode_context, ctx.ctx_params = context, ctx_params
        ctx.func, ctx.rtol, ctx.atol, ctx.fwd_method, ctx.backwd_method, ctx.options = func, rtol, atol, fwd_method, backwd_method, options
        setup_context(func, context)
        with torch.no_grad():
            ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=fwd_method, options=options)
        func.reset_context()
        ctx.save_for_backward(t, flat_params, *ans)
        return ans

    @staticmethod
    def backward(ctx, *grad_output):

        t, flat_params, *ans = ctx.saved_tensors
        ans = tuple(ans)
        func, rtol, atol, method, options = ctx.func, ctx.rtol, ctx.atol, ctx.backwd_method, ctx.options        
        setup_context(func, ctx.ode_context)
        n_tensors = len(ans)
        f_params = tuple(func.parameters())

        if ctx.ctx_params is not None:
            ctx_list = []
            for k, v in ctx.ode_context.items():
                if v is not None and v.requires_grad:
                    ctx_list.append(v)
            flat_params = torch.cat([flat_params, ctx.ctx_params])
            f_params = tuple(list(f_params) + ctx_list)

        # TODO: use a nn.Module and call odeint_adjoint to implement higher order derivatives.
        def augmented_dynamics(t, y_aug):
            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.
            y, adj_y = y_aug[:n_tensors], y_aug[n_tensors:2 * n_tensors]  # Ignore adj_time and adj_params.

            with torch.set_grad_enabled(True):
                t = t.to(y[0].device).detach().requires_grad_(True)
                y = tuple(y_.detach().requires_grad_(True) for y_ in y)
                func_eval = func(t, y)
                vjp_t, *vjp_y_and_params = torch.autograd.grad(
                    func_eval, (t,) + y + f_params,
                    tuple(-adj_y_ for adj_y_ in adj_y), allow_unused=True, retain_graph=True
                )
            vjp_y = vjp_y_and_params[:n_tensors]
            vjp_params = vjp_y_and_params[n_tensors:]

            # autograd.grad returns None if no gradient, set to zero.
            vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
            vjp_y = tuple(torch.zeros_like(y_) if vjp_y_ is None else vjp_y_ for vjp_y_, y_ in zip(vjp_y, y))
            vjp_params = _flatten_convert_none_to_zeros(vjp_params, f_params)

            if len(f_params) == 0:
                vjp_params = torch.tensor(0.).to(vjp_y[0])
            return (*func_eval, *vjp_y, vjp_t, vjp_params)

        T = ans[0].shape[0]
        with torch.no_grad():
            adj_y = tuple(grad_output_[-1] for grad_output_ in grad_output)
            adj_params = torch.zeros_like(flat_params)
            adj_time = torch.tensor(0.).to(t)
            time_vjps = []
            for i in range(T - 1, 0, -1):

                ans_i = tuple(ans_[i] for ans_ in ans)
                grad_output_i = tuple(grad_output_[i] for grad_output_ in grad_output)
                func_i = func(t[i], ans_i)

                # Compute the effect of moving the current time measurement point.
                dLd_cur_t = sum(
                    torch.dot(func_i_.view(-1), grad_output_i_.view(-1)).view(1)
                    for func_i_, grad_output_i_ in zip(func_i, grad_output_i)
                )
                adj_time = adj_time - dLd_cur_t
                time_vjps.append(dLd_cur_t)

                # Run the augmented system backwards in time.
                if len(adj_params) == 0:
                    adj_params = torch.tensor(0.).to(adj_y[0])
                aug_y0 = (*ans_i, *adj_y, adj_time, adj_params)
                aug_ans = odeint(
                    augmented_dynamics, aug_y0,
                    torch.tensor([t[i], t[i - 1]]), rtol=rtol, atol=atol, method=method, options=options
                )

                # Unpack aug_ans.
                adj_y = aug_ans[n_tensors:2 * n_tensors]
                adj_time = aug_ans[2 * n_tensors]
                adj_params = aug_ans[2 * n_tensors + 1]

                adj_y = tuple(adj_y_[1] if len(adj_y_) > 0 else adj_y_ for adj_y_ in adj_y)
                if len(adj_time) > 0: adj_time = adj_time[1]
                if len(adj_params) > 0: adj_params = adj_params[1]

                adj_y = tuple(adj_y_ + grad_output_[i - 1] for adj_y_, grad_output_ in zip(adj_y, grad_output))

                del aug_y0, aug_ans

            time_vjps.append(adj_time)
            time_vjps = torch.cat(time_vjps[::-1])
            func.reset_context()
            grad_ctx = None
            if ctx.ctx_params is not None:
                adj_params, grad_ctx = torch.split(adj_params, [adj_params.shape[0] - ctx.ctx_params.shape[0], 
                                                                ctx.ctx_params.shape[0]])                
            return (*adj_y, None, grad_ctx, None, time_vjps, adj_params, None, None, None, None, None)


def cond_odeint_adjoint(func, y0, t, context=None, rtol=1e-6, atol=1e-12, 
                        fwd_method=None, backwd_method=None, options=None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(func, nn.Module):
        raise ValueError('func is required to be an instance of nn.Module.')

    tensor_input = False
    if torch.is_tensor(y0):

        class TupleFunc(nn.Module):

            def __init__(self, base_func):
                super(TupleFunc, self).__init__()
                self.base_func = base_func

            def forward(self, t, y):
                return (self.base_func(t, y[0]),)

        tensor_input = True
        y0 = (y0,)
        func = TupleFunc(func)

    flat_params = _flatten(func.parameters())
    ctx_params = None
    if context is not None:
        ctx_param_list = []
        for k, v in context.items():
            if v is not None and v.requires_grad:
                ctx_param_list.append(v.contiguous().view(-1))
        if len(ctx_param_list):
            ctx_params = torch.cat(ctx_param_list)
    ys = CondOdeintAdjointMethod.apply(*y0, context, ctx_params, func, t, flat_params, rtol, atol, 
                                       fwd_method, backwd_method, options)

    if tensor_input:
        ys = ys[0]
    return ys


def cond_odeint(func, y0, t, context=None, rtol=1e-6, atol=1e-12,
                fwd_method=None, backwd_method=None, options=None):
    setup_context(func, context)
    assert fwd_method == backwd_method
    ans = odeint(func, y0, t, rtol, atol, fwd_method, options)
    func.reset_context()
    return ans