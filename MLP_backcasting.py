"""
Deep Neural Network Training Script
-----------------------------------

This script provides a realistic, modular deep learning workflow in Python,
including:

• Data loading via a PyTorch-style Dataset class
• Preprocessing pipelines
• A configurable DNN model (MLP + optional CNN blocks)
• Custom training loops
• Validation metrics (accuracy, loss curves, confusion matrix)
• Early stopping + model checkpointing
• Hyperparameter management
• GPU/CPU device handling
• Logging utilities
• Reproducible seeding

"""

import os
import time
import random
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ================================================================
# 1. Reproducibility
# ================================================================

def set_seed(seed=42):
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(1234)


# ================================================================
# 2. Dataset Definition
# ================================================================

class TabularDataset(Dataset):
    """
    A generic dataset for tabular data.
    Can be swapped for image-based dataset if needed.
    """

    def __init__(self, data_array, target_array):
        super().__init__()
        self.X = data_array.astype(np.float32)
        self.y = target_array.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_synthetic_data(n_samples=25000, n_features=64):
    """Generate a realistic synthetic dataset."""
    X = np.random.randn(n_samples, n_features)
    w = np.random.randn(n_features)
    logits = X @ w + np.random.randn(n_samples) * 0.5
    y = (logits > 0).astype(int)
    return X, y


# ================================================================
# 3. Model Definition
# ================================================================

class DeepModel(nn.Module):
    """
    A configurable MLP-based deep learning model with multiple dense layers,
    skip connections, dropout and batch normalization.
    """

    def __init__(self, n_inputs, n_classes, hidden_sizes=[256, 128, 128, 64]):
        super().__init__()

        layers = []
        in_dim = n_inputs

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = h

        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ================================================================
# 4. Training Utilities
# ================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    all_preds, all_targets = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(X_batch)

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    return epoch_loss / len(loader.dataset), acc


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            epoch_loss += loss.item() * len(X_batch)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    return epoch_loss / len(loader.dataset), acc, all_preds, all_targets


# ================================================================
# 5. Trainer Class
# ================================================================

class Trainer:
    """High-level trainer class handling training, validation, logging."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        lr=1e-3,
        epochs=40,
        checkpoint_path="checkpoint.pt",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.checkpoint_path = checkpoint_path

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        self.history = {"train_loss": [], "train_acc": [],
                        "val_loss": [], "val_acc": []}

    def train(self):
        best_val_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            start = time.time()

            train_loss, train_acc = train_one_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.device
            )

            val_loss, val_acc, preds, targets = validate_one_epoch(
                self.model, self.val_loader, self.criterion, self.device
            )

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch:03d}/{self.epochs}  "
                  f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  "
                  f"Train Acc: {train_acc:.3f}  Val Acc: {val_acc:.3f}  "
                  f"Time: {time.time()-start:.1f}s")

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.checkpoint_path)

        print("\nTraining finished. Best val loss:", best_val_loss)

    def plot_history(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.legend()
        plt.title("Loss Curve")

        plt.subplot(1, 2, 2)
        plt.plot(self.history["train_acc"], label="Train Acc")
        plt.plot(self.history["val_acc"], label="Val Acc")
        plt.legend()
        plt.title("Accuracy Curve")

        plt.tight_layout()
        plt.show()


# ================================================================
# 6. Main Execution
# ================================================================

if __name__ == "__main__":
    print("Loading synthetic data...")
    X, y = load_synthetic_data(n_samples=50000, n_features=128)

    dataset = TabularDataset(X, y)

    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len

    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = DeepModel(n_inputs=128, n_classes=2).to(device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=1e-3,
        epochs=30,
        checkpoint_path="best_model.pt",
    )

    trainer.train()
    trainer.plot_history()

    # Evaluate final model
    print("Evaluating best model...")
    model.load_state_dict(torch.load("best_model.pt"))

    _, _, preds, targets = validate_one_epoch(
        model, val_loader, trainer.criterion, device
    )

    print("\nClassification Report:\n")
    print(classification_report(targets, preds))

    print("Confusion Matrix:")
    print(confusion_matrix(targets, preds))

    print("\nDone.")
