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
                grad_sqrt = torch.sign(p.grad) * torch.sqrt(torch.abs(p.grad))

                # Lazy initialization of momentum buffer
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(p.data)
                    state['prev_grad'] = grad_sqrt

                # Momentum update 
                buf = state['buf']
                mask = torch.where((buf * p.grad > 0), grad_sqrt.abs()/state['prev_grad'], 1 - grad_sqrt.abs()/state['prev_grad'])
                buf.mul_(mask).add_(grad_sqrt)

                # Core SGD update: w = w - lr * grad
                p.add_(buf, alpha=-group['lr'])
                
                state['prev_grad'] = grad_sqrt.abs()

        return loss