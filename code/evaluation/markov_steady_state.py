import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from utils import ttc_ranges


def _fmt_bound(x: float) -> str:
    if np.isneginf(x):
        return "-inf"
    if np.isposinf(x):
        return "inf"
    if float(x).is_integer():
        return str(int(x))
    return str(float(x)).rstrip("0").rstrip(".")


def _range_to_label(rng: Tuple[float, float]) -> str:
    return f"{_fmt_bound(rng[0])}-{_fmt_bound(rng[1])}"


def _parse_tuple_range(key: str) -> Tuple[float, float]:
    s = key.strip()
    def _parse_bound(token: str) -> float:
        t = token.strip()
        if t == "inf":
            return float("inf")
        if t == "-inf":
            return -float("inf")
        return float(t)

    if not (s.startswith("(") and s.endswith(")")):
        raise ValueError(f"Unsupported TTC key format: {key}")
    inside = s[1:-1]
    parts = inside.split(",")
    if len(parts) != 2:
        raise ValueError(f"Unsupported TTC tuple format: {key}")
    a = _parse_bound(parts[0])
    b = _parse_bound(parts[1])
    return float(a), float(b)


def _canonical_label(
    key: str,
    allowed_by_range: Dict[Tuple[float, float], str],
    allowed_labels: set,
) -> str:
    s = key.strip()
    s_no_space = s.replace(" ", "")
    if s in allowed_labels:
        return s
    if s_no_space in allowed_labels:
        return s_no_space
    rng = _parse_tuple_range(s)
    if rng not in allowed_by_range:
        raise ValueError(f"TTC bucket {rng} from data is not defined in utils.ttc_ranges.")
    return allowed_by_range[rng]


def _build_transition_matrix(
    transition_data: Dict[str, Dict[str, float]]
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    ordered_ranges = [(float(a), float(b)) for a, b in ttc_ranges]
    labels = [_range_to_label(rng) for rng in ordered_ranges]
    allowed = {rng: _range_to_label(rng) for rng in ordered_ranges}
    allowed_labels = set(labels)
    label_to_idx = {lb: i for i, lb in enumerate(labels)}
    n = len(labels)
    # Default missing rows to self-loop, so P is always row-stochastic.
    P = np.eye(n, dtype=float)
    has_observed_row = np.zeros((n,), dtype=bool)

    for raw_from, raw_tos in transition_data.items():
        from_label = _canonical_label(raw_from, allowed, allowed_labels)
        i = label_to_idx[from_label]
        row = np.zeros((n,), dtype=float)
        for raw_to, value in raw_tos.items():
            to_label = _canonical_label(raw_to, allowed, allowed_labels)
            j = label_to_idx[to_label]
            row[j] += float(value)
        row_sum = row.sum()
        if row_sum > 0:
            row = row / row_sum
            has_observed_row[i] = True
        else:
            # Keep chain valid even when a row is empty in collected transitions.
            row[i] = 1.0
        P[i, :] = row

    # Default regenerative rule requested by pipeline:
    # crash bucket (-inf, 0) transitions to the initial safe bucket (highest TTC) with prob 1.
    crash_label = _range_to_label((float("-inf"), 0.0))
    init_label = _range_to_label((float(ttc_ranges[-1][0]), float(ttc_ranges[-1][1])))
    if crash_label in label_to_idx and init_label in label_to_idx:
        ci = label_to_idx[crash_label]
        ii = label_to_idx[init_label]
        P[ci, :] = 0.0
        P[ci, ii] = 1.0
        has_observed_row[ci] = True

    return P, labels, has_observed_row


def compute_steady_state_from_transition(
    transition_file: str,
    max_iterations: int = 10000,
    tolerance: float = 1e-12,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, List[str]]:
    transition_path = Path(transition_file)
    with transition_path.open("r") as f:
        transition_data: Dict[str, Dict[str, float]] = json.load(f)

    P, labels, has_observed_row = _build_transition_matrix(transition_data)
    n = len(labels)

    def _power_iteration() -> Tuple[np.ndarray, np.ndarray]:
        # Start from the practical initial bucket used in this pipeline:
        # the highest TTC range (e.g., "8-inf"), not a uniform prior over all buckets.
        x = np.zeros((n,), dtype=float)
        x[-1] = 1.0
        errors = np.zeros((max_iterations,), dtype=float)
        for iteration in range(max_iterations):
            x_new = x @ P
            err = float(np.max(np.abs(x_new - x)))
            errors[iteration] = err
            x = x_new
            if err < tolerance:
                return x, errors[: iteration + 1]
        return x, errors

    def _exact_solve() -> np.ndarray:
        # Solve [P^T - I; 1...1] x = [0...0; 1]
        A = P.T - np.eye(n, dtype=float)
        A[-1, :] = 1.0
        b = np.zeros((n,), dtype=float)
        b[-1] = 1.0
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        # Numerical cleanup
        x = np.where(np.abs(x) < 1e-15, 0.0, x)
        x = np.maximum(x, 0.0)
        s = x.sum()
        if s <= 0:
            raise np.linalg.LinAlgError("Invalid stationary solution with non-positive sum.")
        return x / s

    # Exact solve is only meaningful when all rows have observed transition data.
    # With empty/missing rows, chain is reducible by construction and power iteration
    # from the intended initial bucket is the correct semantics for this pipeline.
    if n < 10 and bool(np.all(has_observed_row)):
        try:
            x = _exact_solve()
            errors = np.array([], dtype=float)
        except np.linalg.LinAlgError:
            x, errors = _power_iteration()
    else:
        x, errors = _power_iteration()

    steady_state = {labels[i]: float(x[i]) for i in range(n)}
    crash_label = _range_to_label((float("-inf"), 0.0))
    crash_rate = steady_state.get(crash_label, 0.0)
    return crash_rate, steady_state, errors, P, labels


def save_results(
    transition_file: str,
    crashrate: float,
    steady_state_map: Dict[str, float],
) -> Path:
    output_dir = Path(transition_file).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "steady_state_result.json"
    payload = {
        "crash_rate": crashrate,
        "steady_state": steady_state_map,
    }
    with result_path.open("w") as f:
        json.dump(payload, f, indent=4)
    return result_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python evaluation/markov_steady_state.py <accumulated_transition_counts.json>")
    crashrate, steady_state_map, _, _, _ = compute_steady_state_from_transition(sys.argv[1])
    save_results(sys.argv[1], crashrate, steady_state_map)
    print(crashrate)
