import torch
import math

def newton_schulz_orthogonalize(M, steps=7, eps=1e-7):
    assert len(M.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = M
    X /= (X.norm() + eps)
    if M.size(0) > M.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if M.size(0) > M.size(1):
        X = X.T
    return X.to(M.dtype)

class SimpleOrthoOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-2, momentum=0.95, nesterov=True, ns_steps=6,
                 adamw_lr=1e-3, adamw_betas=(0.95, 0.95), adamw_eps=1e-8, adamw_wd=0,
                 min_dim=2):

        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_lr=adamw_lr, adamw_betas=adamw_betas,
                        adamw_eps=adamw_eps, adamw_wd=adamw_wd,
                        min_dim=min_dim)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']
            weight_decay = group['adamw_wd']
            min_dim = group['min_dim']
            lr_ratio = group['adamw_lr'] / lr

            for p in group['params']:
                g = p.grad
                if g is None:
                    continue

                if p.ndim >= min_dim and p.size(0) < 10000:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    
                    original_shape = p.data.shape # Store the parameter's original shape
                    p_flat = p.data.view(p.data.size(0), -1).clone() # Flatten the parameter
                    g_flat = g.view(g.size(0), -1) # Flatten the gradient

                    g_flat = newton_schulz_orthogonalize(g_flat, steps=ns_steps)
                    g_flat *= max(1, g_flat.size(0) / g_flat.size(1)) ** 0.5

                    p_flat.add_(g_flat, alpha=-lr)  # Add update to flattened parameter
                    p.data = p_flat.view(original_shape) # Reshape and assign back to p.data
                else:
                    state = self.state[p]
                    if 'step' not in state:
                        state['step'] = 0
                        state['moment1'] = torch.zeros_like(g)
                        state['moment2'] = torch.zeros_like(g)
                    state['step'] += 1
                    step = state['step']
                    buf1 = state['moment1']
                    buf2 = state['moment2']
                    buf1.lerp_(g, 1 - beta1)
                    buf2.lerp_(g.square(), 1 - beta2)

                    g = buf1 / (eps + buf2.sqrt())

                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    scale = bias_correction1 / bias_correction2 ** 0.5
                    p.data.mul_(1 - lr_ratio * lr * weight_decay)
                    p.data.add_(g, alpha=-lr_ratio*lr/scale)
        return loss