import torch
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tqdm
import typer

class TestFunction:
    def __init__(self, name, func, local_minima=None, global_minima=None, x_range=(-5, 5), y_range=(-5, 5), starting_point=None):
        self.name = name
        self.func = func
        self.local_minima = local_minima if local_minima else []  # Default to empty list
        self.global_minima = global_minima if global_minima else [] # Default to empty list
        self.x_range = x_range
        self.y_range = y_range
        self.starting_point = starting_point if starting_point else [0.0, 0.0] # default to (0,0)

    def __call__(self, tensor):
        return self.func(tensor)

    def plot(self, x_values, y_values, losses, scheduler_str): # Method to create and log plots
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        x_range = np.arange(self.x_range[0], self.x_range[1], 0.1)
        y_range = np.arange(self.y_range[0], self.y_range[1], 0.1)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[self(torch.tensor([x_val, y_val])).item() for x_val in x_range] for y_val in y_range])


        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

        ax.plot(x_values, y_values, losses, marker='o', linestyle='-', color='red', label='Optimization Path', linewidth=1, markersize=4, zorder=10)

        # Mark minima
        for x, y, z in self.local_minima:
            ax.plot(x, y, z, marker='^', color='blue', label='Local Minima', zorder=12)
        for x, y, z in self.global_minima:
            ax.plot(x, y, z, marker='*', color='green', label='Global Minima', zorder=12)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Loss')
        ax.set_title(f'Optimization Path on {self.name} ({scheduler_str})')
        plt.legend()


        wandb.log({f"{scheduler_str}_plot": wandb.Image(plt)})
        plt.close(fig)

# Define test functions as instances of the TestFunction class
rosenbrock = TestFunction("rosenbrock", lambda tensor: 10 * (1 - tensor[0]) ** 2 + 100 * (tensor[1] - tensor[0] ** 2) ** 2, global_minima=[(1.0, 1.0, 0.0)], x_range=(-3, 2), y_range=(-3, 4))
ackley = TestFunction("ackley", lambda tensor: 1.06674770338 * (-20 * torch.exp(-0.2 * torch.sqrt(0.5 * (tensor[0] ** 2 + tensor[1] ** 2))) - torch.exp(0.5 * (torch.cos(2 * math.pi * tensor[0]) + torch.cos(2 * math.pi * tensor[1]))) + math.e + 20), global_minima=[(0.0, 0.0, 0.0)], starting_point=[-4., -2.])
rastrigin = TestFunction("rastrigin", lambda tensor: 0.5 * (20 + tensor[0] ** 2 - 10 * torch.cos(2 * math.pi * tensor[0]) + tensor[1] ** 2 - 10 * torch.cos(2 * math.pi * tensor[1])), global_minima=[(0.0, 0.0, 0.0)], starting_point=[-4., -2.])
michalewicz = TestFunction("michalewicz", lambda tensor: 5.55154 * (1.8013 - (torch.sin(tensor[0]) * torch.sin(tensor[0]**2 / math.pi)**20 + torch.sin(tensor[1]) * torch.sin(2*tensor[1]**2/math.pi)**20)), global_minima=[(2.20, 1.57, 0.0)], starting_point=[-2., -2.])  # Approximate global minimum
easom = TestFunction("easom", lambda tensor: 10 * (1 - torch.cos(tensor[0]) * torch.cos(tensor[1]) * torch.exp(-((tensor[0] - math.pi)**2 + (tensor[1] - math.pi)**2))), global_minima=[(math.pi, math.pi, 0.0)], x_range=(-0, 6), y_range=(-1, 7))
booth = TestFunction("booth", lambda tensor: 0.135135135135 * (tensor[0] + 2*tensor[1] - 7)**2 + (2*tensor[0] + tensor[1] - 5)**2, global_minima=[(1.0, 3.0, 0.0)], x_range=(-5, 10), y_range=(-5, 10))
himmelblau = TestFunction("himmelblau", lambda tensor: 0.0588235294118 * (tensor[0]**2 + tensor[1] - 11)**2 + (tensor[0] + tensor[1]**2 - 7)**2, global_minima=[(3.0, 2.0, 0.0), (-2.805118, 3.131312, 0.0), (-3.779310, -3.283186, 0.0), (3.584428, -1.848126, 0.0)]) # added global minima for more clarity
levi_n13 = TestFunction("levi_n13", lambda tensor: 5 * (torch.sin(3*math.pi*tensor[0])**2 + (tensor[0]-1)**2 * (1 + torch.sin(3*math.pi*tensor[1])**2) + (tensor[1]-1)**2 * (1 + torch.sin(2*math.pi*tensor[1])**2)), global_minima=[(1.0, 1.0, 0.0)])
schaffer_n2 = TestFunction("schaffer_n2", lambda tensor: 10.9289617486 * (0.5 + (torch.sin(tensor[0]**2 - tensor[1]**2)**2 - 0.5) / (1 + 0.001*(tensor[0]**2 + tensor[1]**2))**2), global_minima=[(0.0, 0.0, 0.0)], starting_point=[-4., -2.])
shekel = TestFunction("shekel", lambda tensor: 1.0480697176 * (10.1532 - sum(1 / ( (tensor[0] - a[0])**2 + (tensor[1] - a[1])**2 + c  ) for a, c in zip([[4, 4], [1, 1], [8, 8], [6, 6], [3, 7], [2, 9], [5, 5], [8, 1], [6, 2], [7, 3.6]], [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]))), global_minima=[(4.0, 4.0, 0.0)], x_range=(0, 10), y_range=(0, 10))

test_functions = {func.name: func for func in [rosenbrock, ackley, rastrigin, michalewicz, easom, booth, himmelblau, levi_n13, schaffer_n2, shekel]}


def optimization_test_task(optimizer_class, optimizer_config, wandb_project="optimization-test"):

    optimizer_name = optimizer_class.__name__
    hyperparam_str = "-".join(f"{k}={v}" for k, v in optimizer_config["hyperparameters"].items())
    wandb_run_name = f"{optimizer_name}-{hyperparam_str}-all_functions"  # Single run name

    wandb.init(project=wandb_project, name=wandb_run_name, config=optimizer_config)

    num_steps = optimizer_config.get("steps", 300)
    warmup_steps = int(optimizer_config.get("warmup_percentage", 0.05) * num_steps)
    cooldown_steps = int(optimizer_config.get("cooldown_percentage", 0.35) * num_steps)
    
    cum_loss = 0

    for test_function in test_functions.values():  # Iterate through all functions

        for use_scheduler in [False, True]:  # Test with and without scheduler
            # Initialize tensor
            x = torch.tensor(test_function.starting_point, requires_grad=True)

            # Optimizer
            optimizer = optimizer_class([x], **optimizer_config["hyperparameters"])

            # Scheduler (Conditional)
            if use_scheduler:
                lr_lambda = lambda step: max(0., min(1., (num_steps - step) / cooldown_steps) )  if step >= num_steps - cooldown_steps else min((step+1) / warmup_steps, 1.0)
                scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            else:
                scheduler = None

            scheduler_str = "with_scheduler" if use_scheduler else "no_scheduler"
            losses = []
            x_values = []
            y_values = []

            for step in tqdm.tqdm(range(num_steps), desc=f"Optimizing {test_function.name} ({scheduler_str})", unit="step"):
                optimizer.zero_grad()
                loss = test_function(x)
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                    wandb.log({f"{test_function.name}_step": step + 1, f"{test_function.name}_loss": loss.item(), f"{test_function.name}_lr": current_lr, f"{test_function.name}_x": x[0].item(), f"{test_function.name}_y": x[1].item()})
                else:
                    wandb.log({f"{test_function.name}_step": step + 1, f"{test_function.name}_loss": loss.item(), f"{test_function.name}_x": x[0].item(), f"{test_function.name}_y": x[1].item()})

                losses.append(loss.item())
                x_values.append(x[0].item())
                y_values.append(x[1].item())

            test_function.plot(x_values, y_values, losses, scheduler_str)

            if losses[-1] < 1e8:
              cum_loss += losses[-1]
            wandb.run.summary[f"{test_function.name}_{scheduler_str}_final_loss"] = losses[-1]
    
    wandb.run.summary["cum_final_loss"] = cum_loss

    wandb.finish()  # Close wandb run


def main(lr: float = 0.02, wd: float = 0., b1: float = 0.9, b2: float = 0.99):
    optimizer_config = {
        "optimizer": optim.AdamW,  # Pass the class, not a string
        "hyperparameters": {
            "lr": lr,
            "weight_decay": wd,
            "betas": (b1, b2),
        },
        "warmup_percentage": 0.05,  # Example customization
        "cooldown_percentage": 0.3

    }
    
    optimization_test_task(optimizer_config["optimizer"], optimizer_config)

if __name__ == "__main__":
    typer.run(main)