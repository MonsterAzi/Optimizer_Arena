import torch.optim as optim
from image_class import image_classification_task
from optimization_test import optimization_test_task
from simple_2nd import SimpleOrthoOptimizer

# Example optimizer configurations
optimizer_config = {
    "optimizer": SimpleOrthoOptimizer,  # Pass the class, not a string
    "hyperparameters": {
        "lr": 3e-1,
        "momentum": 0.95,
        "nesterov": True,
        "ns_steps": 6,
        "adamw_lr": 1,
        "adamw_betas": (0.95, 0.99),
        "adamw_eps": 1e-8,
        "adamw_wd": 0
    },
    "batch_size": 1024,
    "epochs": 1,
    "warmup_percentage": 0.05,  # Example customization
    "cooldown_percentage": 0.3

}


optimization_test_task(optimizer_config["optimizer"], optimizer_config)