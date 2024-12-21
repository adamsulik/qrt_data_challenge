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

def choose_device(device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)
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
    def __init__(self, input_dim, hidden_layers=[64, 32, 16], dropout_rate=[0.4, 0.3, 0.2]):
        """
        Deep Neural Network for Binary Classification

        Args:
            input_dim (int): Number of input features
            hidden_layers (list): Number of neurons in each hidden layer
            dropout_rate (list): Dropout rate for each hidden layer
        """
        super(DeepNeuralNetwork, self).__init__()

        layers = []
        layer_sizes = [input_dim] + hidden_layers

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate[i]))

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


def date_based_kfold_split(n_splits, dates, shuffle=True):
    """
    Create K-Fold splits based on dates
    """
    # Get unique dates and sort them
    unique_dates = np.unique(dates)
    unique_dates.sort()

    # Split the unique dates into n_splits
    dates_splits = np.array_split(unique_dates, n_splits)

    for i in range(len(dates_splits)):
        # Validation dates for the current split
        val_dates = dates_splits[i]

        # Training dates are all the dates before the current validation dates
        remaining_splits = np.delete(np.arange(n_splits), i)
        train_dates = np.concatenate([dates_splits[j] for j in remaining_splits])

        # Create masks to get train and test indices
        train_indices = np.where(np.isin(dates, train_dates))[0]
        val_indices = np.where(np.isin(dates, val_dates))[0]
        if shuffle:
            np.random.shuffle(train_indices)

        yield train_indices, val_indices


import numpy as np


def date_based_train_test_split(dates, test_size=0.2, shuffle=True):
    """
    Create a train-test split based on unique dates

    Parameters:
    -----------
    dates : array-like
        Array of dates corresponding to each data point
    test_size : float, optional (default=0.2)
        Proportion of unique dates to use for testing
    shuffle : bool, optional (default=True)
        Whether to shuffle the training indices

    Returns:
    --------
    train_indices : numpy.ndarray
        Indices for the training set
    test_indices : numpy.ndarray
        Indices for the test set
    """
    # Get unique dates and sort them
    unique_dates = np.unique(dates)
    unique_dates.sort()

    # Calculate the number of test dates
    n_test_dates = max(1, int(len(unique_dates) * test_size))

    # Split the unique dates into train and test
    test_dates = unique_dates[-n_test_dates:]
    train_dates = unique_dates[:-n_test_dates]

    # Create masks to get train and test indices
    train_indices = np.where(np.isin(dates, train_dates))[0]
    test_indices = np.where(np.isin(dates, test_dates))[0]

    # Shuffle training indices if requested
    if shuffle:
        np.random.shuffle(train_indices)

    return train_indices, test_indices


def run_k_fold_training(X, y, n_splits=4, epochs=50, batch_size=32,
                        random_seed=42, print_every_epoch=10,
                        num_workers=0, device=None, lr=0.001,
                        model_hidden_layers=[64, 32, 16], dropout_rate=[0.4, 0.3, 0.2],
                        split_type='stratified', date_array=None):
    """
    Perform K-Fold Cross-Validation

    Args:
        X (numpy.ndarray): Input features
        y (numpy.ndarray): Binary labels
        n_splits (int): Number of cross-validation splits
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        random_seed (int): Random seed for reproducibility
        print_every_epoch (int): Print metrics every n epochs
        num_workers (int): Number of workers for data loading
        device (str): Device to train on (cpu, cuda, mps)
        lr (float): Learning rate for optimizer
        model_hidden_layers (list): List of hidden layer sizes
        dropout_rate (float): Dropout rate for regularization
        split_type (str): Type of split (stratified, normal) or date based split
        date_array (numpy.ndarray): Array of dates for date-based split

    Returns:
        list: Cross-validation results
    """
    assert split_type in ['stratified', 'normal', 'date'], "Invalid split type"
    _set_seed(random_seed)

    # choose device cuda - mps - cpu
    device = choose_device(device)

    # if dropout rate is a single value, convert to list
    if not isinstance(dropout_rate, list):
        dropout_rate = [dropout_rate] * len(model_hidden_layers)

    print(f"Training on {device}")
    if split_type != 'date':
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        split = skf.split(X, y)
    else:
        split = date_based_kfold_split(n_splits, dates=date_array, shuffle=True)


    cv_results = []

    for fold, (train_index, val_index) in enumerate(split, 1):
        print(f"\nFold {fold}/{n_splits}")
        print(f"Train/Val ratio: {len(train_index)/len(val_index):.2f}")
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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

        # Initialize model
        model = DeepNeuralNetwork(
            hidden_layers=model_hidden_layers,
            input_dim=X_train.shape[1],
            dropout_rate=dropout_rate,
        ).to(device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_acc = 0
        best_metrics = None

        # Training loop
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
            epoch_val = evaluate_model(model, val_loader, criterion, device)
            val_acc = epoch_val['accuracy']
            if val_acc > best_val_acc:
                best_metrics = epoch_val
                best_metrics['epoch'] = epoch
                best_metrics['train_loss'] = train_loss
                best_metrics['train_acc'] = train_acc
                best_metrics['fold'] = fold
                best_val_acc = val_acc

            if epoch % print_every_epoch == 0:
                # print with time
                current_strf_time = time.strftime("%H:%M:%S", time.localtime())
                print(f"{current_strf_time}\t  -- Epoch {epoch}: "
                      f"Training Loss = {train_loss:.4f}"
                      f" | Training Accuracy = {train_acc:.4f}"
                      f" | Validation Accuracy = {val_acc:.4f}")

        print(f'Computation time: {time.time() - start_time:.2f} seconds')

        cv_results.append(best_metrics)

        print("\nValidation Metrics:")
        for metric, value in best_metrics.items():
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
        seed: int = 42,
        print_every_epoch: int = 10,
        num_workers=0, device=None, lr=0.001,
        model_hidden_layers=[64, 32, 16], dropout_rate=0.3,
        split_type='stratified', date_array=None,
        crit_metric_name = 'accuracy',
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
        print_every_epoch (int): Print metrics every n epochs
        num_workers (int): Number of workers for data loading
        device (str): Device to train on (cpu, cuda, mps)
        lr (float): Learning rate for optimizer
        model_hidden_layers (list): List of hidden layer sizes
        dropout_rate (float): Dropout rate for regularization
        split_type (str): Type of split (stratified, normal) or date based split, options:
        ar['stratified', 'normal', 'date']
        date_array (numpy.ndarray): Array of dates for date-based split
        crit_metric_name (str): Metric to use for checkpointing

    Returns:
        dict: Best model checkpoint
    """
    assert split_type in ['stratified', 'normal', 'date'], "Invalid split type"
    # Ensure reproducibility
    _set_seed(seed)

    # Determine device
    device = choose_device(device)

    # Split data into train and validation sets
    if split_type != 'date':
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=val_split, random_state=seed, stratify=Y
        )
    else:
        train_idx, val_idx = date_based_train_test_split(date_array, test_size=val_split)
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize model
    model = DeepNeuralNetwork(
        hidden_layers=model_hidden_layers,
        input_dim=X_train.shape[1],
        dropout_rate=dropout_rate,
    ).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training setup
    best_crit_metric = 0
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
        crit_metric = val_metrics[crit_metric_name]


        # Update best model if loss improves
        if crit_metric > best_crit_metric:
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
            best_crit_metric = crit_metric

        # Store training and validation metrics
        train_loss_arr.append(train_loss)
        train_acc_arr.append(train_acc)
        val_loss_arr.append(val_loss)
        val_acc_arr.append(val_acc)

        # Print and save periodically
        if epoch % print_every_epoch == 0:
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
        batch_size=32,
        model_hidden_layers=[64, 32, 16],
):
    """
    Load saved model and run inference on evaluation dataset

    Args:
        checkpoint_path (str/Path): Path to model checkpoint
        X_eval (numpy.ndarray): Evaluation features
        numerical_columns (list): Names of numerical columns for scaling
        batch_size (int): Inference batch size
        model_hidden_layers (list): List of hidden layer

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
    model = DeepNeuralNetwork(input_dim=X_eval.shape[1], hidden_layers=model_hidden_layers).to(device)
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