import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import numba
import time
import pandas as pd

from .nn_utils import DeepNeuralNetwork, train_model, evaluate_model, choose_device


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


class AugmentedBinaryClassificationDataset(Dataset):
    def __init__(self, df, feature_columns, target_column, augmented_num=3):
        """
        Custom PyTorch dataset with data augmentation based on date-based sampling

        Args:
            df (pd.DataFrame): Input dataframe
            feature_columns (list): Numerical feature columns to use
            target_column (str): Target column name
        """
        df = df.reset_index(drop=True)

        # Prepare features and targets
        self.feature_columns = feature_columns
        self.features = df[feature_columns].values
        self.targets = df[target_column].values

        self.augmented_num = augmented_num

        # Group by date to enable date-based sampling
        self.date_groups = df.groupby('DATE')
        self.dates = df['DATE'].values

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Get original sample features
        original_features = self.features[idx]
        target = self.targets[idx]

        # Find entries with the same date
        current_date = self.dates[idx]
        date_group = self.date_groups.get_group(current_date)

        # Randomly select 3 additional entries from the same date
        additional_indices = np.random.choice(
            date_group.index,
            size=min(self.augmented_num, len(date_group)),
            replace=False
        )

        # Collect additional features
        additional_features = self.features[additional_indices]

        # Combine features
        combined_features = np.concatenate([
            original_features,
            additional_features.flatten(),
        ])

        return (
            torch.tensor(combined_features, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )


def prepare_augmented_dataset(df, augmented_num, feature_columns):
    """
    Prepare augmented dataset with specified configurations

    Args:
        df (pd.DataFrame): Input dataframe
        augmented_num (int): Number of augmented entries to be put into each sample
        feature_columns (list): List of feature columns to use

    Returns:
        AugmentedBinaryClassificationDataset: Augmented dataset
    """

    # Create augmented dataset
    augmented_dataset = AugmentedBinaryClassificationDataset(
        df,
        feature_columns=feature_columns,
        target_column='RET',
        augmented_num=augmented_num
    )

    return augmented_dataset


# Example of creating data loaders with augmentation
def create_augmented_data_loaders(train_df, shuffle, augmented_num, feature_columns, batch_size=32):
    """
    Create train and validation data loaders with augmentation

    Args:
        train_df (pd.DataFrame): Training dataframe
        shuffle (bool): Whether to shuffle the data
        augmented_num (int): Number of augmented entries to be put into each sample
        feature_columns (list): List of feature columns to use
        batch_size (int): Batch size for data loaders

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Assuming train_test_split has been done beforehand
    augmented_train_dataset = prepare_augmented_dataset(train_df, augmented_num, feature_columns)

    train_loader = DataLoader(
        augmented_train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return train_loader


def run_augmented_k_fold_training(
        df,
        augment_num=3,
        n_splits=4,
        epochs=50,
        batch_size=32,
        random_seed=42,
        model_hidden_layers=[124, 64, 32, 16],
):
    """
    Perform K-Fold Cross-Validation with Data Augmentation

    Args:
        df (pd.DataFrame): Input dataframe containing features and target
        n_splits (int): Number of cross-validation splits
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        random_seed (int): Random seed for reproducibility
        model_hidden_layers (list): List of hidden layer sizes

    Returns:
        list: Cross-validation results
    """
    # Set seed for reproducibility
    _set_seed(random_seed)

    # Determine device
    device = choose_device()
    print(f"Training on {device}")

    feature_columns = [col for col in df.columns if col not in ['RET', 'DATE']]

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    X = df[[c for c in df.columns if c not in ['RET', 'DATE']]]
    y = df['RET']
    cv_results = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        start_time = time.time()
        current_strf_time = time.strftime("%H:%M:%S", time.localtime())
        print(f'{current_strf_time}\t -- Started training fold {fold}')

        # Split data
        train_df = df.iloc[train_index].reset_index(drop=True)
        val_df = df.iloc[val_index].reset_index(drop=True)

        # Create augmented train dataset and loader
        data_loader_creator = create_augmented_data_loaders
        train_loader = data_loader_creator(train_df,
                                           shuffle=True,
                                           batch_size=batch_size,
                                           augmented_num=augment_num,
                                           feature_columns=feature_columns)
        val_loader = data_loader_creator(val_df,
                                         shuffle=False,
                                         batch_size=batch_size,
                                         augmented_num=augment_num,
                                         feature_columns=feature_columns)

        # Initialize model
        input_dim = len(feature_columns) * (augment_num+1)
        model = DeepNeuralNetwork(input_dim=input_dim, hidden_layers=model_hidden_layers).to(device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)

            if epoch % 10 == 0:
                # Print with time
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
