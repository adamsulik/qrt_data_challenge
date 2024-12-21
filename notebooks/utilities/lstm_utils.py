import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt

try:
    from .nn_utils import choose_device
except ImportError:
    from nn_utils import choose_device

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class TimeSeriesPredictionDataset(Dataset):
    """
    Custom PyTorch Dataset for time series prediction with additional attributes.

    Args:
        time_series (torch.Tensor): Input time series sequences
        attributes (torch.Tensor): Additional contextual attributes
        targets (torch.Tensor): Target values for prediction
    """

    def __init__(
            self,
            time_series: torch.Tensor,
            attributes: torch.Tensor,
            targets: torch.Tensor
    ):
        # Validate input dimensions
        assert time_series.size(0) == attributes.size(0) == targets.size(0), \
            "Batch dimensions must match across inputs"

        self.time_series = time_series
        self.attributes = attributes
        self.targets = targets

    def __len__(self) -> int:
        return self.time_series.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.time_series[idx],
            self.attributes[idx],
            self.targets[idx]
        )


class MultiInputTimeSeriesModel(nn.Module):
    """
    Multi-input neural network for time series prediction.

    Architecture:
    - LSTM for time series processing
    - Separate path for additional attributes
    - Feature fusion with configurable layers

    Args:
        time_series_features (int): Number of features in time series input
        num_attributes (int): Number of additional attributes
        hidden_sizes (List[int]): Hidden layer sizes for time series and attribute paths
        lstm_hidden_size (int): LSTM hidden state size
        num_lstm_layers (int): Number of LSTM layers
        dropout_rate (float): Dropout probability for regularization
    """

    def __init__(
            self,
            time_series_features: int,
            num_attributes: int,
            hidden_sizes: List[int] = [64, 32],
            lstm_hidden_size: int = 64,
            num_lstm_layers: int = 2,
            dropout_rate: float = 0.3
    ):
        super().__init__()

        # Time series LSTM processor
        self.lstm = nn.LSTM(
            input_size=time_series_features,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate
        )

        # Attributes processing network
        attr_layers = []
        attr_input_size = num_attributes
        for hidden_size in hidden_sizes:
            attr_layers.extend([
                nn.Linear(attr_input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            attr_input_size = hidden_size

        self.attribute_processor = nn.Sequential(*attr_layers)

        # Fusion and output layers
        fusion_input_size = lstm_hidden_size + attr_input_size
        output_layers = []
        output_input_size = fusion_input_size

        for hidden_size in hidden_sizes:
            output_layers.extend([
                nn.Linear(output_input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            output_input_size = hidden_size

        output_layers.append(nn.Linear(output_input_size, 1))
        self.output_network = nn.Sequential(*output_layers)

    def forward(
            self,
            time_series: torch.Tensor,
            attributes: torch.Tensor
    ) -> torch.Tensor:
        # Process time series with LSTM
        lstm_out, _ = self.lstm(time_series)
        lstm_last_output = lstm_out[:, -1, :]  # Take last time step

        # Process additional attributes
        attr_processed = self.attribute_processor(attributes)

        # Fuse features
        fused_features = torch.cat([lstm_last_output, attr_processed], dim=1)

        # Final prediction
        prediction = self.output_network(fused_features)
        return prediction


class TimeSeriesTrainer:
    """
    Comprehensive training pipeline for time series prediction models.

    Handles training, validation, evaluation, and visualization.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: torch.device,
            learning_rate: float = 1e-3,
            patience: int = 10,
            checkpoint_dir: str = 'checkpoints',
            sign_penalty = 1.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.sign_penalty = sign_penalty

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        self.mse_loss = nn.MSELoss()

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.early_stopping = EarlyStopping(patience=patience)

    def sign_aware_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Custom loss function that combines MSE with a sign prediction penalty.

        Args:
            predictions: Model predictions
            targets: Ground truth values

        Returns:
            Combined loss value
        """
        # Basic MSE loss
        mse_loss = self.mse_loss(predictions, targets)

        # Sign penalty: Binary cross entropy on the sign
        pred_signs = torch.sign(predictions)
        target_signs = torch.sign(targets)

        # Convert signs to binary (1 for positive, 0 for negative/zero)
        pred_signs = (pred_signs > 0).float()
        target_signs = (target_signs > 0).float()

        # Calculate sign loss using binary cross entropy
        sign_loss = nn.BCEWithLogitsLoss()(pred_signs, target_signs)

        # Combine losses
        total_loss = mse_loss + self.sign_penalty * sign_loss

        return total_loss

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for time_series, attributes, targets in self.train_loader:
            time_series = time_series.to(self.device)
            attributes = attributes.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(time_series, attributes)
            loss = self.sign_aware_loss(predictions, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for time_series, attributes, targets in self.val_loader:
                time_series = time_series.to(self.device)
                attributes = attributes.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(time_series, attributes)
                loss = self.sign_aware_loss(predictions, targets)

                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = total_loss / len(self.val_loader)
        metrics = self.compute_metrics(
            np.array(all_targets),
            np.array(all_predictions)
        )

        return val_loss, metrics

    def train(
            self,
            max_epochs: int = 100,
            print_every: int = 1
    ) -> Dict[str, List[float]]:
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'mae': [],
            'rmse': [],
            'r2': []
        }

        for epoch in range(1, max_epochs+1):
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()

            self.scheduler.step(val_loss)

            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['mae'].append(metrics['mae'])
            training_history['rmse'].append(metrics['rmse'])
            training_history['r2'].append(metrics['r2'])

            if epoch % print_every == 0:
                print(f"Epoch {epoch}/{max_epochs}")
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print(f"MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}\n")

            # Early stopping and model checkpointing
            if self.early_stopping(val_loss):
                print("Early stopping triggered.")
                break

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss
            }, os.path.join(self.checkpoint_dir, f'best_model_epoch_{epoch}.pth'))

        return training_history

    @staticmethod
    def compute_metrics(
            targets: np.ndarray,
            predictions: np.ndarray
    ) -> Dict[str, float]:
        mae = np.mean(np.abs(targets - predictions))
        rmse = np.sqrt(np.mean((targets - predictions) ** 2))
        r2 = 1 - (np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2))

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

    def plot_learning_curves(self, history: Dict[str, List[float]]):
        plt.figure(figsize=(15, 5))
        plt.title('Training & Validation Loss')
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting.

    Tracks validation loss and stops training if no improvement is observed.
    """

    def __init__(self, patience: int = 10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def generate_synthetic_data(
        num_samples: int = 1000,
        sequence_length: int = 30,
        time_series_features: int = 5,
        num_attributes: int = 3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic time series data for demonstration.

    Returns:
        time_series, attributes, targets
    """
    time_series = torch.randn(num_samples, sequence_length, time_series_features)
    attributes = torch.randn(num_samples, num_attributes)

    # Simulated target with some complexity
    targets = (
            time_series.mean(dim=(1, 2)) * 0.5 +
            attributes.sum(dim=1) * 0.3 +
            torch.randn(num_samples) * 0.2
    ).unsqueeze(1)

    return time_series, attributes, targets


def predict(
        model: torch.nn.Module,
        time_series: Union[torch.Tensor, np.ndarray],
        attributes: Union[torch.Tensor, np.ndarray],
        device: Optional[torch.device] = None,
        return_details: bool = False,
        batch_size: int = 32
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Generate predictions using a multi-input time series model.

    Args:
        model (torch.nn.Module): Trained PyTorch model
        time_series (Union[torch.Tensor, np.ndarray]): Input time series sequences
        attributes (Union[torch.Tensor, np.ndarray]): Additional contextual attributes
        device (Optional[torch.device]): Device to run inference on (CPU/GPU)
        return_details (bool): If True, returns additional internal representations
        batch_size (int): Number of samples to process in each batch

    Returns:
        predictions (torch.Tensor): Model predictions
        Optional additional details if return_details is True
    """
    # Set default device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert inputs to tensors if they're numpy arrays
    if isinstance(time_series, np.ndarray):
        time_series = torch.from_numpy(time_series).float()
    if isinstance(attributes, np.ndarray):
        attributes = torch.from_numpy(attributes).float()

    # Validate input dimensions
    assert time_series.ndim == 3, "Time series must be 3D: (batch, sequence, features)"
    assert attributes.ndim == 2, "Attributes must be 2D: (batch, features)"
    assert time_series.size(0) == attributes.size(0), "Batch sizes must match"

    # Prepare model for inference
    model.eval()
    model.to(device)

    # Prepare output storage
    all_predictions = []

    # Optional: Store additional details
    if return_details:
        internal_details = {
            'lstm_outputs': [],
            'attribute_processed': [],
            'fused_features': []
        }

    # Inference with batching to handle large datasets
    with torch.no_grad():
        for i in range(0, len(time_series), batch_size):
            # Prepare batch
            batch_time_series = time_series[i:i + batch_size].to(device)
            batch_attributes = attributes[i:i + batch_size].to(device)

            # Forward pass
            predictions = model(batch_time_series, batch_attributes)

            # Optional: Capture internal representations
            if return_details:
                # Hook into the model to capture internal representations
                # Note: This requires modifying the forward method to support this
                # You might need to add a custom method in your model class
                lstm_out, _ = model.lstm(batch_time_series)
                lstm_last_output = lstm_out[:, -1, :]

                attr_processed = model.attribute_processor(batch_attributes)
                fused_features = torch.cat([lstm_last_output, attr_processed], dim=1)

                internal_details['lstm_outputs'].append(lstm_last_output.cpu())
                internal_details['attribute_processed'].append(attr_processed.cpu())
                internal_details['fused_features'].append(fused_features.cpu())

            # Store predictions
            all_predictions.append(predictions.cpu())

    # Combine predictions
    final_predictions = torch.cat(all_predictions, dim=0)

    # Return with optional details
    if return_details:
        # Combine internal details
        internal_details = {
            k: torch.cat(v, dim=0)
            for k, v in internal_details.items()
        }
        return final_predictions, internal_details

    return final_predictions


def main():
    # Set up device
    device = choose_device()
    print(f"Using device: {device}")

    # Generate synthetic data
    time_series, attributes, targets = generate_synthetic_data()
    print(time_series.shape)
    print('---')
    print(attributes.shape)
    print('---')
    print(targets.shape)
    print('---')

    # Create dataset and split
    dataset = TimeSeriesPredictionDataset(time_series, attributes, targets)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize model
    model = MultiInputTimeSeriesModel(
        time_series_features=time_series.shape[2],
        num_attributes=attributes.shape[1]
    ).to(device)

    # Training pipeline
    trainer = TimeSeriesTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # Train and get history
    training_history = trainer.train(max_epochs=50)

    # Plot learning curves
    trainer.plot_learning_curves(training_history)

    print("Training complete. Learning curves saved to 'learning_curves.png'")


if __name__ == '__main__':
    main()