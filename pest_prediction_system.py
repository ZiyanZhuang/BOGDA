"""
BOGDA -Pest Resistance Prediction Model with Temporal Convolutional Network and Attention Mechanism
Authors: Ziyan Zhuang, Qiao Sheng, Jingang Xie
License: MIT
"""

import datetime
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import norm
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Global Configuration
CONFIG = {
    "data_path": "data.xlsx",  # Path to experimental data (sensitive - not included in repo)
    "seq_length": 5,  # Time window for temporal features
    "batch_size": 32,  # Training batch size
    "epochs": 200,  # Maximum training epochs
    "pred_steps": 25,  # Future generations to predict
    "num_channels": [128, 128, 128],  # TCN layer configurations
    "slope_reg_weight": 0.5  # Regularization strength for resistance slope
}


# Data Processor
class LC50Processor:
    """Processes raw bioassay data to calculate resistance ratios"""

    def __init__(self):
        self._cache = {}  # Cache for memoization

    def _safe_log(self, x):
        """Numerically stable logarithmic transformation"""
        return np.log(np.clip(x, 1e-10, None))

    def _probit_fit(self, concentrations, mortality):
        """Probit analysis for LC50 calculation using curve fitting"""
        try:
            # Data validation and cleaning
            valid = (concentrations > 0) & (~np.isnan(concentrations)) & (~np.isnan(mortality))
            concentrations = concentrations[valid]
            mortality = mortality[valid]
            if len(concentrations) < 2:
                return np.nan

            # Nonlinear curve fitting with biological constraints
            popt, _ = curve_fit(
                lambda x, a, b: norm.cdf(a + b * self._safe_log(x)),
                concentrations, mortality,
                bounds=([-np.inf, 0.1], [np.inf, 5]),  # Realistic parameter bounds
                maxfev=5000  # Increased iteration limit
            )
            return np.exp((norm.ppf(0.5) - popt[0]) / popt[1])
        except Exception as e:
            print(f"Probit fit failed: {str(e)}")
            return np.nan

    def calculate_LC50(self, df):
        """Main processing method with memoization"""
        cache_key = hash(pd.util.hash_pandas_object(df).sum())
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Group-wise LC50 calculation with error handling
        result = df.groupby('generations', group_keys=False).apply(
            lambda g: self._probit_fit(
                g['concentration(mg/l)'].values,
                g['dead number'].values / g['insects'].values
            )
        )
        # Data imputation for missing values
        self._cache[cache_key] = result.interpolate().ffill().bfill()
        return self._cache[cache_key]


# Neural Network Model
class TCN_Attention(nn.Module):
    """Temporal Convolutional Network with Attention Mechanism

    Architecture:
    - Stacked dilated convolutions for temporal pattern extraction
    - Multi-head attention for feature importance weighting
    - Final regression layer for resistance prediction
    """

    def __init__(self, input_size=1, num_channels=None):
        super().__init__()
        num_channels = num_channels or CONFIG["num_channels"]

        # Temporal Convolutional Layers
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels[0], 3, padding=1),
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU(),
            nn.Conv1d(num_channels[0], num_channels[1], 3, dilation=2, padding=2),
            nn.BatchNorm1d(num_channels[1]),
            nn.ReLU(),
            nn.Conv1d(num_channels[1], num_channels[2], 3, dilation=4, padding=4),
            nn.BatchNorm1d(num_channels[2]),
            nn.ReLU(),
        )

        # Attention Mechanism
        self.attention = nn.MultiheadAttention(num_channels[-1], 4)

        # Regression output
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        """Forward pass with dimension handling"""
        # Dimension adjustments
        if x.dim() == 4:
            x = x.squeeze(1)
        if x.dim() != 3:
            raise ValueError(f"Invalid input dim: {x.dim()}D (expected 3D [batch, seq_len, features])")

        # Temporal feature extraction
        x = x.permute(0, 2, 1)
        tcn_out = self.tcn(x)

        # Attention processing
        attn_in = tcn_out.permute(2, 0, 1)
        attn_out, _ = self.attention(attn_in, attn_in, attn_in)

        return self.fc(attn_out[-1])


# Training System
class PestResistanceSystem:
    """End-to-end training and prediction system"""

    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self._init_data()
        self._init_model()

    def _init_data(self):
        """Data loading and preprocessing pipeline"""
        if not os.path.exists(self.config["data_path"]):
            raise FileNotFoundError(f"Data file missing: {self.config['data_path']}")

        try:
            # Load and validate raw data
            df = pd.read_excel(
                self.config["data_path"],
                engine='openpyxl',
                usecols=['generations', 'concentration(mg/l)', 'dead number', 'insects']
            )
        except KeyError as e:
            raise ValueError(f"Missing required column: {e}")

        # Calculate resistance ratios
        processor = LC50Processor()
        lc50_series = processor.calculate_LC50(df)
        self.ratio = (lc50_series / lc50_series.iloc[0]).values.reshape(-1, 1)

        # Normalization
        self.scaler = MinMaxScaler()
        self.scaled = self.scaler.fit_transform(self.ratio)

        # Create temporal sequences
        X, y = [], []
        for i in range(len(self.scaled) - self.config["seq_length"]):
            X.append(self.scaled[i:i + self.config["seq_length"]])
            y.append(self.scaled[i + self.config["seq_length"]])

        self.X = torch.FloatTensor(np.array(X))
        self.y = torch.FloatTensor(np.array(y))
        print(f"Dataset shape: X={self.X.shape}, y={self.y.shape}")

        # Train-test split
        split = int(0.8 * len(self.X))
        self.train_X, self.test_X = self.X[:split], self.X[split:]
        self.train_y, self.test_y = self.y[:split], self.y[split:]

    def _init_model(self):
        """Model and optimizer initialization"""
        self.model = TCN_Attention().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=5, factor=0.5
        )

    def compute_slope_regularizer(self, outputs):
        """Regularization to prevent negative resistance trends"""
        gradients = torch.diff(outputs, dim=1)
        negative_grads = torch.clamp(gradients, max=0)
        return torch.mean(negative_grads ** 2)

    def train(self):
        """Main training loop with early stopping"""
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(self.train_X, self.train_y),
            batch_size=self.config["batch_size"],
            shuffle=True
        )

        best_loss = float('inf')
        patience = 0
        log_file = 'training_log.txt'

        with open(log_file, 'w') as f:
            f.write("Training Log\n" + "=" * 50 + "\n")

        for epoch in range(self.config["epochs"]):
            self.model.train()
            epoch_loss = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)

                # Loss calculation with regularization
                main_loss = self.criterion(outputs, batch_y)
                reg_loss = self.compute_slope_regularizer(outputs)
                loss = main_loss + self.config["slope_reg_weight"] * reg_loss

                # Backpropagation
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += main_loss.item()

            # Validation and learning rate adjustment
            val_loss = self._validate()
            self.scheduler.step(val_loss)

            # Logging
            log_msg = (f"Epoch {epoch + 1}/{self.config['epochs']} | "
                       f"Train Loss: {epoch_loss / len(train_loader):.4f} | "
                       f"Val Loss: {val_loss:.4f}")

            with open(log_file, 'a') as f:
                f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {log_msg}\n")

            print(log_msg)

            # Early stopping mechanism
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                patience += 1
            if patience >= 10:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    def _validate(self):
        """Validation phase"""
        self.model.eval()
        val_loader = DataLoader(
            torch.utils.data.TensorDataset(self.test_X, self.test_y),
            batch_size=32,
            shuffle=False
        )

        total_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_x)
                total_loss += self.criterion(outputs, batch_y).item()

        return total_loss / len(val_loader)

    def predict(self, steps=None):
        """Make future predictions"""
        self.model.load_state_dict(torch.load("best_model.pth", map_location=self.device))
        self.model.eval()

        steps = steps or self.config["pred_steps"]
        current_seq = self.scaled[-self.config["seq_length"]:].copy()
        current_seq = torch.FloatTensor(current_seq).unsqueeze(0).to(self.device)

        predictions = []
        with torch.no_grad():
            for _ in range(steps):
                pred = self.model(current_seq)
                predictions.append(pred.cpu().numpy().item())
                # Update sequence with new prediction
                next_seq = current_seq[:, 1:, :].clone()
                next_seq = torch.cat([next_seq, pred.unsqueeze(1)], dim=1)
                current_seq = next_seq

        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    def evaluate(self, X, y):
        """Model performance evaluation"""
        self.model.eval()
        loader = DataLoader(
            torch.utils.data.TensorDataset(X, y),
            batch_size=32,
            shuffle=False
        )

        predictions, true_values = [], []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x).cpu().numpy().flatten()
                predictions.extend(outputs)
                true_values.extend(batch_y.numpy().flatten())

        # Inverse normalization
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        true_values = self.scaler.inverse_transform(np.array(true_values).reshape(-1, 1)).flatten()
        return predictions, true_values

    def visualize(self, predictions):
        """Generate prediction visualization"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.ratio, label='Historical RR', marker='o')
        future_gen = range(len(self.ratio), len(self.ratio) + len(predictions))
        plt.plot(future_gen, predictions, 'r--', marker='s', label='Predicted RR')
        plt.title("Pest Resistance Ratio Prediction")
        plt.xlabel("Generations")
        plt.ylabel("Resistance Ratio (RR)")
        plt.legend()
        plt.grid(True)
        plt.savefig("prediction.png", dpi=300, bbox_inches='tight')
        plt.close()


# Main Program 
if __name__ == "__main__":
    # Initialize system
    system = PestResistanceSystem(CONFIG)
    system.train()

    # Generate predictions
    predictions = system.predict()
    print("\nFuture predictions:")
    print(pd.DataFrame(predictions, columns=['Predicted RR']))

    # Model evaluation
    train_pred, train_true = system.evaluate(system.train_X, system.train_y)
    test_pred, test_true = system.evaluate(system.test_X, system.test_y)

    # Generate report
    metrics = {
        "Training MSE": mean_squared_error(train_true, train_pred),
        "Training R²": r2_score(train_true, train_pred),
        "Test MSE": mean_squared_error(test_true, test_pred),
        "Test R²": r2_score(test_true, test_pred)
    }

    report = [
        "=" * 40 + " Model Evaluation Report " + "=" * 40,
        f"Training samples: {len(train_true)}",
        f"Test samples: {len(test_true)}",
        "\nPerformance Metrics:",
        *[f"{k}: {v:.4f}" for k, v in metrics.items()],
        "\nPrediction Info:",
        f"Prediction horizon: {CONFIG['pred_steps']} generations",
        f"Final predicted RR: {predictions[-1][0]:.2f}"
    ]

    # Save outputs
    with open("model_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    pd.DataFrame({
        'Generation': range(len(system.ratio) + 1, len(system.ratio) + 1 + CONFIG["pred_steps"]),
        'Predicted_RR': predictions.flatten()
    }).to_csv('predictions.csv', index=False)

    # Generate visualization
    system.visualize(predictions)
    print("\nExecution completed!")
    print("Results saved to:")
    print("- training_log.txt : Training metrics")
    print("- model_report.txt : Performance analysis")
    print("- predictions.csv  : Numerical predictions")