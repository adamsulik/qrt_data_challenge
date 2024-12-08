import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pprint import pprint


def _set_seed(seed=42):
    """
    Set random seeds for reproducibility across multiple libraries

    Args:
        seed (int): Seed value for random number generators
    """
    # Python's random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # CuDNN (GPU-specific settings)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BinaryClassificationDataset(Dataset):
    def __init__(self, X, y):
        """
        Custom PyTorch dataset for binary classification

        Args:
            X (torch.Tensor): Input features
            y (torch.Tensor): Binary target labels
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layers=[64, 32, 16], dropout_rate=0.3):
        """
        Deep Neural Network for Binary Classification

        Args:
            input_dim (int): Number of input features
            hidden_layers (list): Number of neurons in each hidden layer
            dropout_rate (float): Dropout probability for regularization
        """
        super(DeepNeuralNetwork, self).__init__()

        layers = []
        layer_sizes = [input_dim] + hidden_layers

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        # Final layer for binary classification
        layers.append(nn.Linear(hidden_layers[-1], 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch

    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch.float())

        # compute training accuracy
        predictions = (outputs > 0.5).float()
        total_correct += (predictions == y_batch).float().sum().item()
        total_samples += y_batch.size(0)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # extract relevant values
    avg_loss = total_loss / len(train_loader)
    train_accuracy = total_correct / total_samples

    return avg_loss, train_accuracy


def evaluate_model(model, val_loader, criterion, device):
    """
    Evaluate model performance

    Returns:
        dict: Performance metrics
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch.float())

            preds = (outputs > 0.5).float()

            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(val_loader)

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1_score': f1_score(all_labels, all_preds),
    }

    return metrics


def run_k_fold_training(X, y, n_splits=4, epochs=50, batch_size=32, random_seed=42):
    """
    Perform K-Fold Cross-Validation

    Args:
        X (numpy.ndarray): Input features
        y (numpy.ndarray): Binary labels
        n_splits (int): Number of cross-validation splits
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        random_seed (int): Random seed for reproducibility

    Returns:
        list: Cross-validation results
    """
    _set_seed(random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    cv_results = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}")
        start_time = time.time()

        # Split data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)

        # Create datasets and loaders
        train_dataset = BinaryClassificationDataset(X_train_tensor, y_train_tensor)
        val_dataset = BinaryClassificationDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize model
        model = DeepNeuralNetwork(input_dim=X_train.shape[1]).to(device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)

            if epoch % 10 == 0:
                # print with time
                current_strf_time = time.strftime("%H:%M:%S", time.localtime())
                print(f"{current_strf_time}\t  -- Epoch {epoch}: "
                      f"Training Loss = {train_loss:.4f}"
                      f" | Training Accuracy = {train_acc:.4f}")

        # Evaluate model
        metrics = evaluate_model(model, val_loader, criterion, device)
        metrics['fold'] = fold
        print(f'Computation time: {time.time() - start_time:.2f} seconds')

        cv_results.append(metrics)

        print("\nValidation Metrics:")
        for metric, value in metrics.items():
            if metric != 'fold':
                print(f"{metric}: {value}")

    return cv_results
