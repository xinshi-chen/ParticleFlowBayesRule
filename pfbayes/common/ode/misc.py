from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
from pfbayes.common.ode.ode_nets import TimePredNet, KernelEmbedNet, ContextCondNet
from pfbayes.common.ode.cnf import CNF
from pfbayes.common.ode.odefunc import ODEfunc
from pfbayes.common.ode.ode_nets import ODEnet
from pfbayes.common.ode.ode_layers import SequentialFlow
from pfbayes.common.consts import DEVICE


def set_cnf_options(args, model):

    def _set(module):
        if isinstance(module, CNF):
            # Set training settings
            module.fwd_solver = args.fwd_solver
            module.backwd_solver = args.backwd_solver
            module.atol = args.atol
            module.rtol = args.rtol
            if args.step_size is not None:
                module.solver_options['step_size'] = args.step_size

            # If using fixed-grid adams, restrict order to not be too high.
            if args.fwd_solver in ['fixed_adams', 'explicit_adams']:
                module.solver_options['max_order'] = 4

            # Set the test settings
            module.test_solver = args.test_solver if args.test_solver else args.fwd_solver
            module.test_atol = args.test_atol if args.test_atol else args.atol
            module.test_rtol = args.test_rtol if args.test_rtol else args.rtol

        if isinstance(module, ODEfunc):
            module.rademacher = args.rademacher
            module.residual = args.residual

    model.apply(_set)


def grad_ll(ob, x, ll_func):
    ll = ll_func(ob, x)
    with torch.set_grad_enabled(True):
        grad_x = torch.autograd.grad(ll, x, 
                                     grad_outputs=torch.ones(ll.size()).to(ll),
                                     create_graph=True, only_inputs=True)[0]        
        return grad_x


def build_model(args, x_dim, ob_dim, regularization_fns=None, ll_func=None):    

    def build_cnf(kernel_embed_net, time_pred_net, context_cond_net):
        diffeq = ODEnet(
            args,
            x_dim=x_dim,
            context_cond_net=context_cond_net,
            prior_embed_dim=kernel_embed_net.output_size,
            ob_dim=ob_dim,
        )
        odefunc = ODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            residual=False,
            rademacher=args.rademacher,
        )
        cnf = CNF(
            odefunc=odefunc,
            T=args.time_length,
            time_mode=args.time_mode,
            time_pred_net=time_pred_net,
            kernel_embedding_net=kernel_embed_net,
            regularization_fns=regularization_fns,
            fwd_solver=args.fwd_solver,
            backwd_solver=args.backwd_solver,
        )
        return cnf

    print('building flow with xdim:', x_dim, 'ob_dim:', ob_dim)
    # network for embedding prior (or sequential posterior) distribution
    kernel_embed_net = KernelEmbedNet(input_dim=x_dim,
                                      hidden_dims=args.kernel_embed_dims,
                                      nonlinearity=args.nonlinearity,
                                      trainable=args.train_kernel_embed)
    # network for predicting integration duration
    if args.time_mode == 'adaptive':
        time_pred_net = TimePredNet(prior_embed_dim=kernel_embed_net.output_size,
                                    ob_dim=ob_dim,
                                    hidden_dims=args.time_pred_dims,
                                    nonlinearity=args.nonlinearity,
                                    time_length=args.time_length)
    else:
        time_pred_net = None
    # network that conditions on all the information embeddings
    if ob_dim is None or not args.feed_context_input:
        context_cond_net = None
    else:
        if args.ob_feed == 'ob':
            ob_trans_func = lambda ob, x: ob
        else:
            assert ll_func is not None
            ob_trans_func = lambda ob, x: grad_ll(ob, x, ll_func)
        context_cond_net = ContextCondNet(args, x_dim=x_dim, prior_embed_dim=kernel_embed_net.output_size,
                                          ob_dim=ob_dim, ob_trans_func=ob_trans_func)
    chain = [build_cnf(kernel_embed_net, time_pred_net, context_cond_net) for _ in range(args.num_blocks)]
    model = SequentialFlow(chain)
    set_cnf_options(args, model)

    return model.to(DEVICE)