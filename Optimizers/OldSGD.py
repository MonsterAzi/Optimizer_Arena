import torch

class SGD1951(torch.optim.Optimizer):
    """
    Implements a simple Stochastic Gradient Descent (SGD) optimizer 
    reminiscent of the 1951 style, without momentum or other modern 
    additions.

    Args:
        params (iterable): iterable of parameters to optimize or dicts 
                           defining parameter groups
        lr (float): learning rate
    """

    def __init__(self, params, lr=1e-3):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(SGD1951, self).__init__(params, defaults)

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
                
                # Core SGD update: w = w - lr * grad
                p.add_(p.grad, alpha=-group['lr'])

        return loss