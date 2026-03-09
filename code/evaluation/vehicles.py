from utils import *
from scipy.stats import norm
import nde as nde
from pathlib import Path
import sys
import json
import numpy as np
import torch
import os


# Load car-following predictor from behavior_modeling if available
CF_PREDICTOR = None
SCALING_LAW_A = None
SCALING_LAW_K = None
CF_MODEL_DIR = None

ROOT_DIR = Path(__file__).resolve().parents[1]  # project root (OpenPAV/OpenPAV)
BM_DIR = ROOT_DIR / "behavior_modeling"
if BM_DIR.exists():
    try:
        sys.path.append(str(BM_DIR))
        from model import CarFollowingPredictor  # type: ignore
        _CF_AVAILABLE = True
    except Exception as e:
        print(f"Failed to import CarFollowingPredictor: {e}")
        _CF_AVAILABLE = False
else:
    _CF_AVAILABLE = False


def _load_cf_predictor(model_dir: Path) -> None:
    global CF_PREDICTOR, SCALING_LAW_A, SCALING_LAW_K, CF_MODEL_DIR
    if not _CF_AVAILABLE:
        return
    if CF_MODEL_DIR == str(model_dir) and CF_PREDICTOR is not None:
        return
    CF_MODEL_DIR = str(model_dir)
    CF_PREDICTOR = None
    SCALING_LAW_A = None
    SCALING_LAW_K = None

    calib_path = model_dir / "calibration.json"
    if calib_path.exists():
        with calib_path.open("r") as f:
            calib = json.load(f)
        SCALING_LAW_A = calib.get("a")
        SCALING_LAW_K = calib.get("k")
    if model_dir.exists():
        CF_PREDICTOR = CarFollowingPredictor(exp_dir=str(model_dir), dist="normal", num_feature=3)

class Vehicle:
    def __init__(self, v, s, lane):
        self.velocity = v
        self.space = s  # 到AV的纵向距离, pos_self-pos_av
        self.lane_id = lane  # 0,1,2

    def __str__(self):
        """ 返回适合打印的字符串格式 """
        return f"V(v={self.velocity}, s={self.space}, lane={self.lane_id})"

    def __repr__(self):
        """ 返回适合调试的字符串格式 """
        return f"V(v={self.velocity}, s={self.space}, lane={self.lane_id})"

    def _stochastic_idm_distribution(self, lead_vehicle):
        """Build AV acceleration distribution from stochastic IDM."""
        acc_mu = max(
            a_min,
            min(
                a_max,
                self.IDM_acceleration(self.velocity, lead_vehicle.velocity, lead_vehicle.space, "AV"),
            ),
        )
        spread = max(3.0 * sigma_acc, action_granularity)
        lower = max(a_min, acc_mu - spread)
        upper = min(a_max, acc_mu + spread)
        acc_range = np.arange(lower, upper + action_granularity * 0.5, action_granularity)
        acc_distribution = norm(loc=acc_mu, scale=max(sigma_acc, 1e-6))
        return acc_range, acc_distribution

    def get_action_distributions_AV(self, BVs, mode):
        if mode == "CF":
            # Use learned stochastic car-following model if available
            model_dir_env = os.environ.get("CF_MODEL_DIR")
            if model_dir_env:
                _load_cf_predictor(Path(model_dir_env))
            elif CF_PREDICTOR is None:
                default_exp = ROOT_DIR / "results" / "MicroSimACC_FAV0_seq1"
                _load_cf_predictor(default_exp)

            if CF_PREDICTOR is not None:
                # Features: [Spatial_Gap, Speed_LV, Speed_FAV]
                spatial_gap = BVs.space
                speed_lv = BVs.velocity
                speed_fav = self.velocity
                x = np.array([[spatial_gap, speed_lv, speed_fav]], dtype=np.float32)  # (1, 3)
                X_tensor = torch.from_numpy(x).unsqueeze(0).float()  # (1, 1, 3)

                mu_arr, sigma_arr, _ = CF_PREDICTOR.predict(X_tensor)
                mu = float(mu_arr[0])
                sigma = float(max(sigma_arr[0], 1e-6))

                # Discrete action space: clamp to global bounds but use [mu-2σ, mu+2σ]
                lower = max(a_min, mu - 2.0 * sigma)
                upper = min(a_max, mu + 2.0 * sigma)
                if lower >= upper:
                    lower = max(a_min, mu - action_granularity)
                    upper = min(a_max, mu + action_granularity)
                acc_range = np.arange(lower, upper + action_granularity * 0.5, action_granularity)

                # Scaling-law based discrete probabilities
                a = getattr(CF_PREDICTOR, "a", None)
                k = getattr(CF_PREDICTOR, "k", None)
                if a is None or k is None:
                    a = SCALING_LAW_A
                    k = SCALING_LAW_K
                if a is None or k is None:
                    raise ValueError("Scaling-law parameters not found in results calibration.json.")
                if a <= 0 or k == 0:
                    raise ValueError("Scaling-law parameters must satisfy a>0 and k!=0.")

                scale = (np.abs(acc_range - mu) / (a * sigma)) + 1.0
                coeff = -1.0 / (2.0 * a * k * sigma)
                probs = coeff * (scale ** (1.0 / k - 1.0))
                probs = np.maximum(probs, 0.0)
                probs = probs * action_granularity
                probs = threshold_and_normalize(probs, distribution_ignore_threshold)

                return acc_range, probs
            else:
                raise RuntimeError("CF predictor is unavailable; check training outputs and CF_MODEL_DIR.")
        elif mode == "IDM":
            return self._stochastic_idm_distribution(BVs)
        else:
            raise ValueError(f"Unsupported AV mode: {mode}")

    def get_action_distributions_BV(self, AV, other_BVs, mode):
        if mode in ("CF", "IDM"):
            if self.velocity > 40:
                acc_lead_distribution = norm(loc=-1, scale=sigma_acc)
                acc_lead_changes = np.arange(-1 - 3 * sigma_acc, -1 + 3 * sigma_acc, action_granularity)
            else:
                acc_lead_distribution = norm(loc=0, scale=sigma_acc)
                acc_lead_changes = np.arange(- 3 * sigma_acc, 3 * sigma_acc, action_granularity)
            return acc_lead_changes, acc_lead_distribution
        else:
            raise ValueError(f"Unsupported BV mode: {mode}")


    @staticmethod
    def IDM_acceleration(v, v_lead, s, vehicle_type):
        COMFORT_ACC_MAX, COMFORT_ACC_MIN, DISTANCE_WANTED, TIME_WANTED, DESIRED_VELOCITY, DELTA = None, None, None, None, None, None
        if vehicle_type == "AV":
            COMFORT_ACC_MAX = CAV_COMFORT_ACC_MAX
            COMFORT_ACC_MIN = CAV_COMFORT_ACC_MIN
            DISTANCE_WANTED = CAV_DISTANCE_WANTED
            TIME_WANTED = CAV_TIME_WANTED
            DESIRED_VELOCITY = CAV_DESIRED_VELOCITY
            DELTA = CAV_DELTA
        elif vehicle_type == "BV":
            COMFORT_ACC_MAX = NDD_COMFORT_ACC_MAX
            COMFORT_ACC_MIN = NDD_COMFORT_ACC_MIN
            DISTANCE_WANTED = NDD_DISTANCE_WANTED
            TIME_WANTED = NDD_TIME_WANTED
            DESIRED_VELOCITY = NDD_DESIRED_VELOCITY
            DELTA = NDD_DELTA

        if v is None:
            return 0
        if v_lead is not None and s is not None:
            s = max(1e-5, abs(s)- car_length)
            s_star = DISTANCE_WANTED + max(0, v * TIME_WANTED + (v * (v - v_lead)) / (
                    2 * np.sqrt(CAV_COMFORT_ACC_MAX * COMFORT_ACC_MIN)))
            acc = max(COMFORT_ACC_MAX * (1 - max((v / DESIRED_VELOCITY) ** DELTA, (s_star / s) ** 2)),#(1 - (v / DESIRED_VELOCITY) ** DELTA-(s_star / s) ** 2),
                      -COMFORT_ACC_MIN)
        else:
            acc = max(COMFORT_ACC_MAX * (1 - (v / DESIRED_VELOCITY) ** DELTA), -COMFORT_ACC_MIN)
        return acc

def generate_initial_state(mode):
    if mode in ("CF", "IDM"):
        FAV = Vehicle(30, 0, 0)
        LBV = Vehicle(30, 47, 0)
        return FAV, LBV
