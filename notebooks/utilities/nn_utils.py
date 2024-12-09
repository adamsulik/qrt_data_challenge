import random
import time
from copy import deepcopy
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pprint import pprint

def choose_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

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


def run_k_fold_training(X, y, n_splits=4, epochs=50, batch_size=32, random_seed=42, print_every_epoch=10):
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

    # choose device cuda - mps - cpu
    device = choose_device()

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

            if epoch % print_every_epoch == 0:
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


def full_dataset_training(
        X: np.ndarray,
        Y: np.ndarray,
        numerical_columns: List[str],
        output_storage_path: Path,
        bestmodel_exp_name: str,
        val_split: float = 0.2,
        batch_size: int = 32,
        epochs: int = 100,
        seed: int = 42
):
    """
    Train on full dataset with checkpointing

    Args:
        X (numpy.ndarray): Input features
        Y (numpy.ndarray): Binary labels
        numerical_columns (list): Names of numerical columns for scaling
        output_storage_path (str/Path): Path to save model checkpoints
        bestmodel_exp_name (str): Name of the checkpoint file
        val_split (float): Validation split ratio
        batch_size (int): Training batch size
        epochs (int): Number of training epochs
        seed (int): Random seed

    Returns:
        dict: Best model checkpoint
    """
    # Ensure reproducibility
    _set_seed(seed)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split data into train and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=val_split, random_state=seed, stratify=Y
    )

    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()

    # Scale only numerical columns
    X_train_scaled[:, :len(numerical_columns)] = scaler.fit_transform(
        X_train[:, :len(numerical_columns)]
    )
    X_val_scaled[:, :len(numerical_columns)] = scaler.transform(
        X_val[:, :len(numerical_columns)]
    )

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(Y_train)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(Y_val)

    # Create datasets and loaders
    train_dataset = BinaryClassificationDataset(X_train_tensor, y_train_tensor)
    val_dataset = BinaryClassificationDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = DeepNeuralNetwork(input_dim=X.shape[1]).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training setup
    best_val_loss = np.inf
    train_loss_arr, train_acc_arr = [], []
    val_loss_arr, val_acc_arr = [], []
    best_model_checkpoint = None

    # Ensure output path exists
    output_storage_path = Path(output_storage_path)
    output_storage_path.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)

        # validate
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_loss, val_acc = val_metrics['loss'], val_metrics['accuracy']

        # Update best model if loss improves
        if val_loss < best_val_loss:
            best_model_checkpoint = {
                'epoch': epoch,
                'loss': train_loss,
                'train_acc': train_acc,
                'scaler': scaler,
                'model_state_dict': deepcopy(model.state_dict()),
                'optimizer_state_dict': deepcopy(optimizer.state_dict()),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }
            best_val_loss = val_loss

        # Store training and validation metrics
        train_loss_arr.append(train_loss)
        train_acc_arr.append(train_acc)
        val_loss_arr.append(val_loss)
        val_acc_arr.append(val_acc)

        # Print and save periodically
        if epoch % 10 == 0:
            current_strf_time = time.strftime("%H:%M:%S", time.localtime())
            print(f"{current_strf_time}\t  -- Epoch {epoch}: \t"
                  f"Training Loss = {train_loss:.4f}"
                  f" | Training Accuracy = {train_acc:.4f} | "
                  f"Validation Loss = {val_loss:.4f} | Validation Accuracy = {val_acc:.4f}")

            # Save checkpoint
            if best_model_checkpoint:
                torch.save(
                    best_model_checkpoint,
                    output_storage_path / f'{bestmodel_exp_name}.pt'
                )

    # Save training and validation metrics
    metrics_df = pd.DataFrame({
        'train_loss': train_loss_arr,
        'train_acc': train_acc_arr,
        'val_loss': val_loss_arr,
        'val_acc': val_acc_arr
    })
    return best_model_checkpoint, metrics_df


def load_model_and_predict(
        checkpoint_path,
        X_eval,
        numerical_columns,
        batch_size=32
):
    """
    Load saved model and run inference on evaluation dataset

    Args:
        checkpoint_path (str/Path): Path to model checkpoint
        X_eval (numpy.ndarray): Evaluation features
        numerical_columns (list): Names of numerical columns for scaling
        batch_size (int): Inference batch size

    Returns:
        numpy.ndarray: Predicted probabilities
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Scale evaluation data using saved scaler
    scaler = checkpoint['scaler']
    X_eval_scaled = X_eval.copy()
    X_eval_scaled[:, :len(numerical_columns)] = scaler.transform(X_eval[:, :len(numerical_columns)])

    # Convert to tensor
    X_eval_tensor = torch.FloatTensor(X_eval_scaled).to(device)

    # Recreate model and load state dict
    model = DeepNeuralNetwork(input_dim=X_eval.shape[1]).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepare evaluation dataset
    eval_dataset = BinaryClassificationDataset(X_eval_tensor, torch.zeros(len(X_eval_tensor)))
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Run inference
    predictions = []
    with torch.no_grad():
        for X_batch, _ in eval_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch).squeeze().cpu().numpy()
            predictions.extend(output)

    return np.array(predictions)