"""
Assignment 3 - MLflow Instrumented PyTorch Training Script
Train a CNN on MNIST with MLflow tracking.

    python train.py --lr 0.001 --batch_size 64 --epochs 5
    python train.py --lr 0.01  --batch_size 128 --epochs 5
    ... (run 5 times with different hyperparameters)
"""

import argparse
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────
# 1. Model Definition
# ──────────────────────────────────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.conv_block(x))


# ──────────────────────────────────────────────
# 2. Training & Evaluation Functions
# ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


# ──────────────────────────────────────────────
# 3. Main Training Script
# ──────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # ── MLflow Setup ───────────────────────────
    mlflow.set_experiment("Assignment3_YosefSelim")   # ← change YourName

    with mlflow.start_run(run_name=f"lr={args.lr}_bs={args.batch_size}"):

        # Pillar 1: Log Parameters
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("batch_size",    args.batch_size)
        mlflow.log_param("epochs",        args.epochs)
        mlflow.log_param("optimizer",     "Adam")
        mlflow.log_param("dropout",       0.3)

        # Pillar 2: Set Tags
        mlflow.set_tag("student_id", "202201255")     # ← change YOUR_ID
        mlflow.set_tag("model",      "SimpleCNN")
        mlflow.set_tag("dataset",    "MNIST")

        # Model, optimizer, loss
        model     = SimpleCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        # Pillar 3: Live Logging inside the training loop
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss,   val_acc   = evaluate(model, test_loader, criterion, device)

            # Log metrics at every epoch (step = epoch number)
            mlflow.log_metric("train_loss",     train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc,  step=epoch)
            mlflow.log_metric("val_loss",       val_loss,   step=epoch)
            mlflow.log_metric("val_accuracy",   val_acc,    step=epoch)

            print(f"Epoch {epoch}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Pillar 4: Save Model (MLflow Model Flavor)
        mlflow.pytorch.log_model(model, artifact_path="model")
        print("Model saved to MLflow artifacts.")


# ──────────────────────────────────────────────
# 4. CLI Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow + PyTorch Training")
    parser.add_argument("--lr",         type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int,   default=64,    help="Batch size")
    parser.add_argument("--epochs",     type=int,   default=5,     help="Number of epochs")
    args = parser.parse_args()
    main(args)
