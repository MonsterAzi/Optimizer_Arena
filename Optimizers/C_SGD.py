import torch

class C_SGD(torch.optim.Optimizer):
    """
    Implements a simple Stochastic Gradient Descent (SGD) optimizer 
    reminiscent of the 1951 style, without momentum or other modern 
    additions.

    Args:
        params (iterable): iterable of parameters to optimize or dicts 
                           defining parameter groups
        lr (float): learning rate
    """

    def __init__(self, params, lr=1e-3, momentum=0.95, eps=1e-8):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError("Invalid learning rate: {}".format(momentum))
        if eps < 0.0:
            raise ValueError("Invalid learning rate: {}".format(eps))

        defaults = dict(lr=lr, momentum=momentum, eps=eps)
        super(C_SGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]

                # Lazy initialization of momentum buffer
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(p.data)

                # Momentum update 
                buf = state['buf']
                buf.mul_(group['momentum']).add_(p.grad)

                mask = (buf * p.grad > 0).to(p.grad.dtype)
                mask = mask * (mask.numel() / (mask.sum() + group['eps']))
                masked_buf = buf * mask

                # Core SGD update: w = w - lr * grad
                p.add_(masked_buf, alpha=-group['lr'])

        return loss