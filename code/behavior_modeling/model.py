import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# Predict the longitudinal and lateral acceleration and corresponding standrad deviation of the ego vehicle using LSTM
class ego_acc_LSTM_dist(nn.Module):
    def __init__(self, num_feature = 5, hidden_size = 128, num_layers = 2, output_size = 4, NUMDIR = 2):
        super(ego_acc_LSTM_dist, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.NUMDIR = NUMDIR # Number of directions (2 for longitudinal and lateral)

        self.lstm = nn.LSTM(num_feature, hidden_size, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, output_size * self.NUMDIR)
        self.fc_sigma = nn.Linear(hidden_size, output_size * self.NUMDIR)


    def forward(self, x): # (B, num_feature, input_size)
        batch_size = x.shape[0]
        # Convert NaN to 0
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  # LSTM layer
        x = out[:, -1, :]
        
        # Mean value
        mu = self.fc_mu(x)  # (B, output_size*2)
        mu = mu.view(batch_size, -1, self.NUMDIR)
        
        # Standard deviation
        sigma = self.fc_sigma(x)  # (B, output_size*2)
        sigma = torch.exp(sigma) + 1e-9
        sigma = sigma.view(batch_size, -1, self.NUMDIR)

        return [mu, sigma]


class CarFollowingPredictor:
    """
    Stochastic car-following prediction model based on a trained LSTM.

    - Loads LSTM weights (best.pth) and calibration.json from an experiment directory.
    - Given a sequence of states, predicts a random acceleration sample for each sequence.

    Expected training directory layout (same as train.py):
        exp/MicroSimACC_FAV{fav_id}_seq{seq_len}/
            best.pth
            calibration.json
    """

    def __init__(
        self,
        exp_dir: str,
        num_feature: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        device: Optional[torch.device] = None,
        dist: str = "normal",
        a: Optional[float] = None,
        k: Optional[float] = None,
    ) -> None:
        """
        Args:
            exp_dir: Directory containing best.pth and calibration.json.
            num_feature: Feature dimension per timestep (default 3: Spatial_Gap, Speed_LV, Speed_FAV).
            hidden_size, num_layers: Must match training hyperparameters.
            device: torch device; defaults to cuda if available else cpu.
            dist: 'normal' or 'power_law' for sampling.
            a, k: Power-law parameters if dist == 'power_law'.
        """
        self.exp_dir = Path(exp_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dist = dist
        self.a = a
        self.k = k

        if self.dist not in ("normal", "power_law"):
            raise ValueError("dist must be 'normal' or 'power_law'")

        # Load model weights
        model_path = self.exp_dir / "best.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"best.pth not found in {self.exp_dir}")

        self.model = ego_acc_LSTM_dist(
            num_feature=num_feature,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
            NUMDIR=1,
        ).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Load calibration parameters (optional)
        calib_path = self.exp_dir / "calibration.json"
        self.bias = 0.0
        self.sigma_scale = 1.0
        if calib_path.exists():
            with calib_path.open("r") as f:
                calib = json.load(f)
            self.bias = float(calib.get("bias", 0.0))
            self.sigma_scale = float(calib.get("sigma_scale", 1.0))
            if self.a is None and "a" in calib:
                self.a = float(calib.get("a"))
            if self.k is None and "k" in calib:
                self.k = float(calib.get("k"))

    def _sample_normal(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return np.random.normal(mu, sigma)

    def _sample_power_law(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Inverse-CDF sampling for power-law tails, following utils.sample_acc.sam_acc_power_law.
        """
        if self.a is None or self.k is None:
            raise ValueError("Power-law sampling requires parameters a and k.")

        accs = []
        for m, s in zip(mu, sigma):
            x_inv = np.random.uniform()
            y_inv = m + self.a * s * np.sign(x_inv - 0.5) * (np.abs(2 * x_inv - 1) ** self.k - 1)
            accs.append(y_inv)
        return np.asarray(accs, dtype=np.float32)

    def predict(
        self,
        X: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict acceleration distribution and sample a random acceleration.

        Args:
            X: Tensor of shape (N, seq_len, num_feature).

        Returns:
            mu: mean acceleration, shape (N,)
            sigma: std acceleration, shape (N,)
            acc_sample: sampled acceleration, shape (N,)
        """
        X = X.to(self.device)
        with torch.no_grad():
            mu_t, sigma_t = self.model(X)
            # Shapes: (N, 1, 1) for our setup
            mu = mu_t.view(-1).cpu().numpy()
            sigma = sigma_t.view(-1).cpu().numpy()

        # Apply calibration
        mu = mu + self.bias
        sigma = sigma * self.sigma_scale

        if self.dist == "normal":
            acc_sample = self._sample_normal(mu, sigma)
        else:
            acc_sample = self._sample_power_law(mu, sigma)

        return mu, sigma, acc_sample


__all__ = ["ego_acc_LSTM_dist", "CarFollowingPredictor"]



