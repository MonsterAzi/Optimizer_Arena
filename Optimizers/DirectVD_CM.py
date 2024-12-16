import torch

class DirectVD_CM(torch.optim.Optimizer):
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
        super(DirectVD_CM, self).__init__(params, defaults)

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
                
                grad_norm = torch.nn.functional.normalize(p.grad, dim=0)
                print(p.grad)
                
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(p.data)

                # Momentum update 
                buf = state['buf']
                mask = (buf * p.grad > 0).to(p.grad.dtype)
                buf.mul_(mask).add_(grad_norm)
                
                # Core SGD update: w = w - lr * grad
                p.add_(buf, alpha=-group['lr'])

        return loss