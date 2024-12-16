import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from torch.optim.lr_scheduler import LambdaLR
import math
from collections import defaultdict
import tqdm  # Import tqdm for progress bar
import typer

# Define the custom model
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

def conv(ch_in, ch_out):
    return nn.Conv2d(ch_in, ch_out, kernel_size=3, padding='same', bias=False)

def make_net():
    act = lambda: nn.GELU()
    bn = lambda ch: nn.BatchNorm2d(ch)
    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=2, padding=0, bias=True),
            act(),
        ),
        nn.Sequential(
            conv(24, 64),
            nn.MaxPool2d(2),
            bn(64), act(),
            conv(64, 64),
            bn(64), act(),
        ),
        nn.Sequential(
            conv(64, 256),
            nn.MaxPool2d(2),
            bn(256), act(),
            conv(256, 256),
            bn(256), act(),
        ),
        nn.Sequential(
            conv(256, 256),
            nn.MaxPool2d(2),
            bn(256), act(),
            conv(256, 256),
            bn(256), act(),
        ),
        nn.MaxPool2d(3), # Changed to kernel_size=3 stride=1 as kernel_size must be odd and pool size is too large for cifar10
        Flatten(),
        nn.Linear(256, 10, bias=False),
        Mul(1/9),
    )

def image_classification_task(optimizer_class, optimizer_config, wandb_project="image-classification"):
    # Create WandB run name based on optimizer and hyperparameters
    optimizer_name = optimizer_class.__name__
    hyperparam_str = "-".join(f"{k}={v}" for k, v in optimizer_config["hyperparameters"].items())
    wandb_run_name = f"{optimizer_name}-{hyperparam_str}"

    wandb.init(project=wandb_project, name=wandb_run_name, config=optimizer_config)

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Transformations
    transform_train = transforms.Compose([
        transforms.Pad(2, padding_mode='reflect'), # Reflection padding for translation
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Data Loading
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    batch_size = optimizer_config.get("batch_size", 64)  # Default batch size
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)


    # Model
    model = make_net().to(device)

    # Optimizer
    optimizer = optimizer_class(model.parameters(), **optimizer_config["hyperparameters"])

    # Scheduler (WSD)
    num_epochs = optimizer_config.get("epochs", 100) # Default epochs
    # Calculate total steps based on dataset size and batch size
    num_training_samples = len(trainset) # Length of trainset
    num_steps_per_epoch = math.ceil(num_training_samples / batch_size) # Steps per epoch calculation
    total_steps = num_epochs * num_steps_per_epoch

    warmup_steps = int(optimizer_config.get("warmup_percentage", 0.05) * total_steps)
    cooldown_steps = int(optimizer_config.get("cooldown_percentage", 0.35) * total_steps)

    lr_lambda = lambda step: max(0., min(1., (total_steps - step) / cooldown_steps) )  if step >= total_steps - cooldown_steps else min((step+1) / warmup_steps, 1.0)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Loss Function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)


    # Training Loop
    for epoch in tqdm.tqdm(range(num_epochs), desc="Epochs", unit="epoch"):  # Progress bar for epochs
        model.train()
        total_loss = 0  # Accumulate loss for the epoch
        with tqdm.tqdm(trainloader, desc="Training", unit="batch", leave=False) as tepoch: # Progress bar for training batches within epoch. leave=False to overwrite progress bar after epoch.
            for i, (images, labels) in enumerate(tepoch):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]

                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item()) # Show current batch loss

                # WandB Logging (Per Step)
                wandb.log({"epoch": epoch + 1, "step": i+1, "batch_loss": loss.item(), "lr": current_lr})

        avg_loss = total_loss / len(trainloader) # calculate average loss

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total

        # WandB Logging (Per Epoch)
        wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss, "val_accuracy": accuracy}) # log average loss for epoch

    # Log additional stats to WandB
    total_params = sum(p.numel() for p in model.parameters())
    wandb.run.summary["batch_size"] = batch_size
    wandb.run.summary["total_params"] = total_params
    wandb.run.summary["warmup_steps"] = warmup_steps
    wandb.run.summary["cooldown_steps"] = cooldown_steps
    # Close wandb run
    wandb.finish()


def main(lr: float = 0.01):
    from Optimizers.waaah import C_SGD
    optimizer_config = {
        "optimizer": C_SGD,  # Pass the class, not a string
        "hyperparameters": {
            "lr": lr
        },
        "epochs": 3,
        "batch_size": 1024,
        "warmup_percentage": 0.05,  # Example customization
        "cooldown_percentage": 0.3

    }
    
    image_classification_task(optimizer_config["optimizer"], optimizer_config)

if __name__ == "__main__":
    typer.run(main)