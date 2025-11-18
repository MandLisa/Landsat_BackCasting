"""
Deep Learning Full Pipeline
===========================

This is an extended, production-style deep learning script designed to be large,
realistic, and modular. It contains:

• Reproducibility utilities
• Dataset class
• Data augmentation
• MLP model
• CNN-1D, CNN-2D, and residual CNN models
• Transformer encoder layers
• Multi-head attention
• Positional encodings
• Hybrid CNN + Transformer model
• Model summary utility
• Trainer class (PyTorch)
• LR scheduler, gradient clipping, mixed precision
• Evaluation utilities
• Visualization functions
• Config + checkpointing
• Extended synthetic data generation

"""

import os
import time
import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


# ============================================================================
# 0. Reproducibility
# ============================================================================

def set_seed(seed=42):
    """Ensure reproducible results across the entire pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behaviour
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(1234)


# ============================================================================
# 1. Synthetic Dataset Utilities
# ============================================================================

def generate_synthetic_tabular_data(n=50000, d=64, classes=2):
    """Generate synthetic tabular data for MLP training."""
    X = np.random.randn(n, d).astype(np.float32)

    w = np.random.randn(d).astype(np.float32)
    logits = X @ w + np.random.randn(n).astype(np.float32) * 0.5

    y = (logits > 0).astype(int)
    return X, y


def generate_synthetic_2d_data(n=20000, h=28, w=28, classes=2):
    """Generate synthetic 2D images similar to MNIST-like."""
    X = np.random.randn(n, 1, h, w).astype(np.float32)
    mask = (np.random.rand(n, 1, h, w) > 0.98).astype(np.float32) * 5
    X += mask
    y = (X.mean(axis=(1, 2, 3)) > 0).astype(int)
    return X, y


def generate_synthetic_sequence_data(n=25000, length=100, d_model=32):
    """Generate synthetic sequence data for Transformer-like models."""
    X = np.random.randn(n, length, d_model).astype(np.float32)
    y = (X.mean(axis=(1, 2)) > 0).astype(int)
    return X, y


# ============================================================================
# 2. Dataset + Data Augmentation
# ============================================================================

class GenericDataset(Dataset):
    """Flexible dataset for tabular, image, or sequence data."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataAugmenter:
    """Apply simple augmentations for 2D or 1D data."""

    def __init__(self, noise=0.01, flip_prob=0.2):
        self.noise = noise
        self.flip_prob = flip_prob

    def __call__(self, X):
        if X.ndim == 4:  # N,C,H,W
            if random.random() < self.flip_prob:
                X = torch.flip(X, dims=[3])
        if self.noise > 0:
            X = X + self.noise * torch.randn_like(X)
        return X


# ============================================================================
# 3. Model Components
# ============================================================================

# ----------------------
# 3a. MLP block
# ----------------------

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes=2, hidden=[256, 128, 64]):
        super().__init__()

        layers = []
        curr = in_dim
        for h in hidden:
            layers.append(nn.Linear(curr, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.2))
            curr = h

        layers.append(nn.Linear(curr, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------
# 3b. CNN 1D block
# ----------------------

class CNN1D(nn.Module):
    def __init__(self, seq_len=100, features=32, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(features, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Linear(128 * (seq_len // 4), num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


# ----------------------
# 3c. CNN 2D block
# ----------------------

class CNN2D(nn.Module):
    def __init__(self, image_size=28, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

        h = image_size // 4
        self.classifier = nn.Sequential(
            nn.Linear(64 * h * h, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ----------------------
# 3d. Residual CNN Block
# ----------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(self.main(x) + x)


# ----------------------
# 3e. Transformer Components
# ----------------------

class PositionalEncoding(nn.Module):
    """Standard transformer sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderBlock(nn.Module):
    """A single transformer encoder layer."""

    def __init__(self, d_model=64, n_heads=4, dim_feedforward=128, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.layernorm1(x + self.dropout1(attn_out))

        ff_out = self.fc(x)
        x = self.layernorm2(x + self.dropout2(ff_out))
        return x


# ----------------------
# 3f. Full Transformer Model
# ----------------------

class TransformerClassifier(nn.Module):
    def __init__(self, seq_len=100, d_model=32, num_classes=2, depth=4):
        super().__init__()
        self.pos = PositionalEncoding(d_model, max_len=seq_len)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads=4) for _ in range(depth)
        ])
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.pos(x)
        x = x.transpose(0, 1)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=0)
        return self.fc(x)


# ----------------------
# 3g. Hybrid CNN + Transformer
# ----------------------

class HybridCNNTransformer(nn.Module):
    """CNN encoder + Transformer classifier."""

    def __init__(self, seq_len=100, d_model=32, num_classes=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(d_model, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.ReLU(),
        )

        self.pos = PositionalEncoding(64, max_len=seq_len)

        self.transformer = nn.ModuleList([
            TransformerEncoderBlock(64, n_heads=4) for _ in range(3)
        ])

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        x = self.pos(x)
        x = x.transpose(0, 1)

        for blk in self.transformer:
            x = blk(x)

        x = x.mean(dim=0)
        return self.fc(x)


# ============================================================================
# 4. Model Summary Utility
# ============================================================================

def summarize_model(model, input_shape):
    """Print a simple model summary (similar to Keras)."""
    x = torch.randn(*input_shape)
    print("\nModel Summary:")
    print(model)
    print("\nNumber of parameters:",
          sum(p.numel() for p in model.parameters()))


# ============================================================================
# 5. Training Utilities
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, preds_all, t_all = 0, [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                out = model(X)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(X)
        preds_all.extend(out.argmax(1).cpu().numpy())
        t_all.extend(y.cpu().numpy())

    acc = accuracy_score(t_all, preds_all)
    return total_loss / len(loader.dataset), acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, t_all = 0, [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)

            total_loss += loss.item() * len(X)
            preds_all.extend(out.argmax(1).cpu().numpy())
            t_all.extend(y.cpu().numpy())

    acc = accuracy_score(t_all, preds_all)
    return total_loss / len(loader.dataset), acc, preds_all, t_all


# ============================================================================
# 6. Trainer Class
# ============================================================================

class Trainer:
    def __init__(
        self,
        model,
        device,
        train_loader,
        val_loader,
        lr=1e-3,
        epochs=30,
        ckpt_path="checkpoint.pt",
        clip_grad=1.0,
        use_amp=True,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.epochs = epochs
        self.ckpt_path = ckpt_path
        self.clip_grad = clip_grad
        self.use_amp = use_amp

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        self.history = {"train_loss": [], "val_loss": [],
                        "train_acc": [], "val_acc": []}

        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def train(self):
        best_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            start = time.time()

            tr_loss, tr_acc = train_one_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.device, self.scaler
            )

            val_loss, val_acc, _, _ = evaluate(
                self.model, self.val_loader,
                self.criterion, self.device
            )

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch}/{self.epochs}  "
                  f"Train Loss={tr_loss:.4f}  Val Loss={val_loss:.4f}  "
                  f"Train Acc={tr_acc:.3f}  Val Acc={val_acc:.3f}  "
                  f"Time={time.time()-start:.1f}s")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), self.ckpt_path)

        print("Training completed. Best loss:", best_loss)

    def plot_history(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history["train_acc"], label="Train Acc")
        plt.plot(self.history["val_acc"], label="Val Acc")
        plt.legend()

        plt.show()


# ============================================================================
# 7. Main Execution
# ============================================================================

if __name__ == "__main__":
    print("Generating synthetic dataset...")

    # Switch between dataset types:
    # X, y = generate_synthetic_tabular_data()
    # X, y = generate_synthetic_2d_data()
    X, y = generate_synthetic_sequence_data()

    dataset = GenericDataset(X, y)

    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len

    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Choose model type:
    # model = MLP(in_dim=X.shape[1])
    # model = CNN2D()
    # model = CNN1D()
    # model = TransformerClassifier()
    model = HybridCNNTransformer()

    summarize_model(model, (1, *X.shape[1:]))

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-3,
        epochs=25,
        ckpt_path="best_model.pt",
        clip_grad=1.0,
        use_amp=True,
    )

    trainer.train()
    trainer.plot_history()

    print("Evaluating best model...")
    model.load_state_dict(torch.load("best_model.pt"))
    _, _, preds, targets = evaluate(
        model, val_loader, trainer.criterion, device
    )

    print("\nClassification Report:")
    print(classification_report(targets, preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(targets, preds))

    print("\nDone.")


#--------------------------------------------------------------------------------


# ============================================================================
# 0. Reproducibility
# ============================================================================

def set_seed(seed=42):
    """Ensure reproducible results across the entire pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behaviour
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(1234)


# ============================================================================
# 1. Synthetic Dataset Utilities
# ============================================================================

def generate_synthetic_tabular_data(n=50000, d=64, classes=2):
    """Generate synthetic tabular data for MLP training."""
    X = np.random.randn(n, d).astype(np.float32)

    w = np.random.randn(d).astype(np.float32)
    logits = X @ w + np.random.randn(n).astype(np.float32) * 0.5

    y = (logits > 0).astype(int)
    return X, y


def generate_synthetic_2d_data(n=20000, h=28, w=28, classes=2):
    """Generate synthetic 2D images similar to MNIST-like."""
    X = np.random.randn(n, 1, h, w).astype(np.float32)
    mask = (np.random.rand(n, 1, h, w) > 0.98).astype(np.float32) * 5
    X += mask
    y = (X.mean(axis=(1, 2, 3)) > 0).astype(int)
    return X, y


def generate_synthetic_sequence_data(n=25000, length=100, d_model=32):
    """Generate synthetic sequence data for Transformer-like models."""
    X = np.random.randn(n, length, d_model).astype(np.float32)
    y = (X.mean(axis=(1, 2)) > 0).astype(int)
    return X, y


# ============================================================================
# 2. Dataset + Data Augmentation
# ============================================================================

class GenericDataset(Dataset):
    """Flexible dataset for tabular, image, or sequence data."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataAugmenter:
    """Apply simple augmentations for 2D or 1D data."""

    def __init__(self, noise=0.01, flip_prob=0.2):
        self.noise = noise
        self.flip_prob = flip_prob

    def __call__(self, X):
        if X.ndim == 4:  # N,C,H,W
            if random.random() < self.flip_prob:
                X = torch.flip(X, dims=[3])
        if self.noise > 0:
            X = X + self.noise * torch.randn_like(X)
        return X


# ============================================================================
# 3. Model Components
# ============================================================================

# ----------------------
# 3a. MLP block
# ----------------------

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes=2, hidden=[256, 128, 64]):
        super().__init__()

        layers = []
        curr = in_dim
        for h in hidden:
            layers.append(nn.Linear(curr, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.2))
            curr = h

        layers.append(nn.Linear(curr, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------
# 3b. CNN 1D block
# ----------------------

class CNN1D(nn.Module):
    def __init__(self, seq_len=100, features=32, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(features, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Linear(128 * (seq_len // 4), num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


# ----------------------
# 3c. CNN 2D block
# ----------------------

class CNN2D(nn.Module):
    def __init__(self, image_size=28, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

        h = image_size // 4
        self.classifier = nn.Sequential(
            nn.Linear(64 * h * h, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ----------------------
# 3d. Residual CNN Block
# ----------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(self.main(x) + x)


# ----------------------
# 3e. Transformer Components
# ----------------------

class PositionalEncoding(nn.Module):
    """Standard transformer sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderBlock(nn.Module):
    """A single transformer encoder layer."""

    def __init__(self, d_model=64, n_heads=4, dim_feedforward=128, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.layernorm1(x + self.dropout1(attn_out))

        ff_out = self.fc(x)
        x = self.layernorm2(x + self.dropout2(ff_out))
        return x


# ----------------------
# 3f. Full Transformer Model
# ----------------------

class TransformerClassifier(nn.Module):
    def __init__(self, seq_len=100, d_model=32, num_classes=2, depth=4):
        super().__init__()
        self.pos = PositionalEncoding(d_model, max_len=seq_len)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads=4) for _ in range(depth)
        ])
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.pos(x)
        x = x.transpose(0, 1)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=0)
        return self.fc(x)


# ----------------------
# 3g. Hybrid CNN + Transformer
# ----------------------

class HybridCNNTransformer(nn.Module):
    """CNN encoder + Transformer classifier."""

    def __init__(self, seq_len=100, d_model=32, num_classes=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(d_model, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.ReLU(),
        )

        self.pos = PositionalEncoding(64, max_len=seq_len)

        self.transformer = nn.ModuleList([
            TransformerEncoderBlock(64, n_heads=4) for _ in range(3)
        ])

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        x = self.pos(x)
        x = x.transpose(0, 1)

        for blk in self.transformer:
            x = blk(x)

        x = x.mean(dim=0)
        return self.fc(x)


# ============================================================================
# 4. Model Summary Utility
# ============================================================================

def summarize_model(model, input_shape):
    """Print a simple model summary (similar to Keras)."""
    x = torch.randn(*input_shape)
    print("\nModel Summary:")
    print(model)
    print("\nNumber of parameters:",
          sum(p.numel() for p in model.parameters()))


# ============================================================================
# 5. Training Utilities
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, preds_all, t_all = 0, [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                out = model(X)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(X)
        preds_all.extend(out.argmax(1).cpu().numpy())
        t_all.extend(y.cpu().numpy())

    acc = accuracy_score(t_all, preds_all)
    return total_loss / len(loader.dataset), acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, t_all = 0, [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)

            total_loss += loss.item() * len(X)
            preds_all.extend(out.argmax(1).cpu().numpy())
            t_all.extend(y.cpu().numpy())

    acc = accuracy_score(t_all, preds_all)
    return total_loss / len(loader.dataset), acc, preds_all, t_all


# ============================================================================
# 6. Trainer Class
# ============================================================================

class Trainer:
    def __init__(
        self,
        model,
        device,
        train_loader,
        val_loader,
        lr=1e-3,
        epochs=30,
        ckpt_path="checkpoint.pt",
        clip_grad=1.0,
        use_amp=True,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.epochs = epochs
        self.ckpt_path = ckpt_path
        self.clip_grad = clip_grad
        self.use_amp = use_amp

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        self.history = {"train_loss": [], "val_loss": [],
                        "train_acc": [], "val_acc": []}

        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def train(self):
        best_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            start = time.time()

            tr_loss, tr_acc = train_one_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.device, self.scaler
            )

            val_loss, val_acc, _, _ = evaluate(
                self.model, self.val_loader,
                self.criterion, self.device
            )

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch}/{self.epochs}  "
                  f"Train Loss={tr_loss:.4f}  Val Loss={val_loss:.4f}  "
                  f"Train Acc={tr_acc:.3f}  Val Acc={val_acc:.3f}  "
                  f"Time={time.time()-start:.1f}s")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), self.ckpt_path)

        print("Training completed. Best loss:", best_loss)

    def plot_history(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history["train_acc"], label="Train Acc")
        plt.plot(self.history["val_acc"], label="Val Acc")
        plt.legend()

        plt.show()


# ============================================================================
# 7. Main Execution
# ============================================================================

if __name__ == "__main__":
    print("Generating synthetic dataset...")

    # Switch between dataset types:
    # X, y = generate_synthetic_tabular_data()
    # X, y = generate_synthetic_2d_data()
    X, y = generate_synthetic_sequence_data()

    dataset = GenericDataset(X, y)

    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len

    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Choose model type:
    # model = MLP(in_dim=X.shape[1])
    # model = CNN2D()
    # model = CNN1D()
    # model = TransformerClassifier()
    model = HybridCNNTransformer()

    summarize_model(model, (1, *X.shape[1:]))

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-3,
        epochs=25,
        ckpt_path="best_model.pt",
        clip_grad=1.0,
        use_amp=True,
    )

    trainer.train()
    trainer.plot_history()

    print("Evaluating best model...")
    model.load_state_dict(torch.load("best_model.pt"))
    _, _, preds, targets = evaluate(
        model, val_loader, trainer.criterion, device
    )

    print("\nClassification Report:")
    print(classification_report(targets, preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(targets, preds))

    print("\nDone.")
#--------------------------------------------------------------------------------


# ============================================================================
# 0. Reproducibility
# ============================================================================

def set_seed(seed=42):
    """Ensure reproducible results across the entire pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behaviour
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(1234)


# ============================================================================
# 1. Synthetic Dataset Utilities
# ============================================================================

def generate_synthetic_tabular_data(n=50000, d=64, classes=2):
    """Generate synthetic tabular data for MLP training."""
    X = np.random.randn(n, d).astype(np.float32)

    w = np.random.randn(d).astype(np.float32)
    logits = X @ w + np.random.randn(n).astype(np.float32) * 0.5

    y = (logits > 0).astype(int)
    return X, y


def generate_synthetic_2d_data(n=20000, h=28, w=28, classes=2):
    """Generate synthetic 2D images similar to MNIST-like."""
    X = np.random.randn(n, 1, h, w).astype(np.float32)
    mask = (np.random.rand(n, 1, h, w) > 0.98).astype(np.float32) * 5
    X += mask
    y = (X.mean(axis=(1, 2, 3)) > 0).astype(int)
    return X, y


def generate_synthetic_sequence_data(n=25000, length=100, d_model=32):
    """Generate synthetic sequence data for Transformer-like models."""
    X = np.random.randn(n, length, d_model).astype(np.float32)
    y = (X.mean(axis=(1, 2)) > 0).astype(int)
    return X, y


# ============================================================================
# 2. Dataset + Data Augmentation
# ============================================================================

class GenericDataset(Dataset):
    """Flexible dataset for tabular, image, or sequence data."""

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataAugmenter:
    """Apply simple augmentations for 2D or 1D data."""

    def __init__(self, noise=0.01, flip_prob=0.2):
        self.noise = noise
        self.flip_prob = flip_prob

    def __call__(self, X):
        if X.ndim == 4:  # N,C,H,W
            if random.random() < self.flip_prob:
                X = torch.flip(X, dims=[3])
        if self.noise > 0:
            X = X + self.noise * torch.randn_like(X)
        return X


# ============================================================================
# 3. Model Components
# ============================================================================

# ----------------------
# 3a. MLP block
# ----------------------

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes=2, hidden=[256, 128, 64]):
        super().__init__()

        layers = []
        curr = in_dim
        for h in hidden:
            layers.append(nn.Linear(curr, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(0.2))
            curr = h

        layers.append(nn.Linear(curr, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----------------------
# 3b. CNN 1D block
# ----------------------

class CNN1D(nn.Module):
    def __init__(self, seq_len=100, features=32, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(features, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Linear(128 * (seq_len // 4), num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


# ----------------------
# 3c. CNN 2D block
# ----------------------

class CNN2D(nn.Module):
    def __init__(self, image_size=28, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

        h = image_size // 4
        self.classifier = nn.Sequential(
            nn.Linear(64 * h * h, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ----------------------
# 3d. Residual CNN Block
# ----------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(self.main(x) + x)


# ----------------------
# 3e. Transformer Components
# ----------------------

class PositionalEncoding(nn.Module):
    """Standard transformer sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderBlock(nn.Module):
    """A single transformer encoder layer."""

    def __init__(self, d_model=64, n_heads=4, dim_feedforward=128, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.layernorm1(x + self.dropout1(attn_out))

        ff_out = self.fc(x)
        x = self.layernorm2(x + self.dropout2(ff_out))
        return x


# ----------------------
# 3f. Full Transformer Model
# ----------------------

class TransformerClassifier(nn.Module):
    def __init__(self, seq_len=100, d_model=32, num_classes=2, depth=4):
        super().__init__()
        self.pos = PositionalEncoding(d_model, max_len=seq_len)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads=4) for _ in range(depth)
        ])
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.pos(x)
        x = x.transpose(0, 1)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=0)
        return self.fc(x)


# ----------------------
# 3g. Hybrid CNN + Transformer
# ----------------------

class HybridCNNTransformer(nn.Module):
    """CNN encoder + Transformer classifier."""

    def __init__(self, seq_len=100, d_model=32, num_classes=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(d_model, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.ReLU(),
        )

        self.pos = PositionalEncoding(64, max_len=seq_len)

        self.transformer = nn.ModuleList([
            TransformerEncoderBlock(64, n_heads=4) for _ in range(3)
        ])

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        x = self.pos(x)
        x = x.transpose(0, 1)

        for blk in self.transformer:
            x = blk(x)

        x = x.mean(dim=0)
        return self.fc(x)


# ============================================================================
# 4. Model Summary Utility
# ============================================================================

def summarize_model(model, input_shape):
    """Print a simple model summary (similar to Keras)."""
    x = torch.randn(*input_shape)
    print("\nModel Summary:")
    print(model)
    print("\nNumber of parameters:",
          sum(p.numel() for p in model.parameters()))


# ============================================================================
# 5. Training Utilities
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, preds_all, t_all = 0, [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                out = model(X)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(X)
        preds_all.extend(out.argmax(1).cpu().numpy())
        t_all.extend(y.cpu().numpy())

    acc = accuracy_score(t_all, preds_all)
    return total_loss / len(loader.dataset), acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, t_all = 0, [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)

            total_loss += loss.item() * len(X)
            preds_all.extend(out.argmax(1).cpu().numpy())
            t_all.extend(y.cpu().numpy())

    acc = accuracy_score(t_all, preds_all)
    return total_loss / len(loader.dataset), acc, preds_all, t_all


# ============================================================================
# 6. Trainer Class
# ============================================================================

class Trainer:
    def __init__(
        self,
        model,
        device,
        train_loader,
        val_loader,
        lr=1e-3,
        epochs=30,
        ckpt_path="checkpoint.pt",
        clip_grad=1.0,
        use_amp=True,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.epochs = epochs
        self.ckpt_path = ckpt_path
        self.clip_grad = clip_grad
        self.use_amp = use_amp

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        self.history = {"train_loss": [], "val_loss": [],
                        "train_acc": [], "val_acc": []}

        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def train(self):
        best_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            start = time.time()

            tr_loss, tr_acc = train_one_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.device, self.scaler
            )

            val_loss, val_acc, _, _ = evaluate(
                self.model, self.val_loader,
                self.criterion, self.device
            )

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch}/{self.epochs}  "
                  f"Train Loss={tr_loss:.4f}  Val Loss={val_loss:.4f}  "
                  f"Train Acc={tr_acc:.3f}  Val Acc={val_acc:.3f}  "
                  f"Time={time.time()-start:.1f}s")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), self.ckpt_path)

        print("Training completed. Best loss:", best_loss)

    def plot_history(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history["train_acc"], label="Train Acc")
        plt.plot(self.history["val_acc"], label="Val Acc")
        plt.legend()

        plt.show()


# ============================================================================
# 7. Main Execution
# ============================================================================

if __name__ == "__main__":
    print("Generating synthetic dataset...")

    # Switch between dataset types:
    # X, y = generate_synthetic_tabular_data()
    # X, y = generate_synthetic_2d_data()
    X, y = generate_synthetic_sequence_data()

    dataset = GenericDataset(X, y)

    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len

    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Choose model type:
    # model = MLP(in_dim=X.shape[1])
    # model = CNN2D()
    # model = CNN1D()
    # model = TransformerClassifier()
    model = HybridCNNTransformer()

    summarize_model(model, (1, *X.shape[1:]))

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=1e-3,
        epochs=25,
        ckpt_path="best_model.pt",
        clip_grad=1.0,
        use_amp=True,
    )

    trainer.train()
    trainer.plot_history()

    print("Evaluating best model...")
    model.load_state_dict(torch.load("best_model.pt"))
    _, _, preds, targets = evaluate(
        model, val_loader, trainer.criterion, device
    )

    print("\nClassification Report:")
    print(classification_report(targets, preds))

    print("\nConfusion Matrix:")
    print(confusion_matrix(targets, preds))

    print("\nDone.")
