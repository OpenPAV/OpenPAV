import os
import sys
import time
import argparse
import math

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils import *  # noqa: F401,F403
import sample_scenarios as sample
import vehicles as veh


print("Current Working Directory:", os.getcwd())
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, "results", "benchmark")

# VT-Micro coefficients: rows=n1 (speed power), cols=n2 (acceleration power)
VTM_COEFF = [
    [-7.537, 0.4438, 0.1716, -0.0420],
    [0.0973, 0.0518, 0.0029, 0.0071],
    [-0.0030, -7.42e-04, 1.09e-04, 1.16e-04],
    [5.3e-05, 6.0e-06, -1.0e-05, -6.0e-06],
]
# Derived from VTM polynomial over current action bounds v in [0, 50], a in [-2, 4]:
# max exp(poly) is around 9.83e3, use a small safety margin.
VTM_FUEL_RATE_MAX = 1.0e4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Output directory for MonteCarlo results")
    parser.add_argument("--iterations", type=int, default=100000, help="Total MonteCarlo iterations")
    parser.add_argument("--interval", type=int, default=10000, help="Logging interval")
    parser.add_argument("--state_recorded", action="store_true", help="Record state distribution")
    parser.add_argument("--no_state_recorded", action="store_true", help="Disable state recording")
    parser.add_argument("--mode", type=str, default="CF", help="Simulation mode")
    return parser.parse_args()


def _vt_micro_fuel_rate(v: float, a: float) -> float:
    poly = 0.0
    for n1 in range(4):
        v_pow = v ** n1
        for n2 in range(4):
            poly += VTM_COEFF[n1][n2] * v_pow * (a ** n2)
    try:
        return float(math.exp(poly))
    except OverflowError:
        return float("inf")


def _compute_time_headway(state, mode):
    if state is None:
        return None
    if mode in ("CF", "IDM"):
        fav = state[0]
        lead = state[1]
        if fav.velocity <= 1e-6:
            return None
        return max(0.0, float(lead.space)) / float(fav.velocity)
    return None


def simulate_monte_carlo(initial_state, iterations, state_recorded, mode, interval, results_dir):
    start_time = time.time()

    critical_state = {}
    ttc_counts = {ttc_range: 0 for ttc_range in ttc_ranges}
    transition_counts = {f"{from_ttc} to {to_ttc}": 0 for from_ttc in ttc_counts for to_ttc in ttc_counts}
    states_prob_record = {}

    crash_log = []
    interval_crash_count = 0

    # Additional statistics
    sum_time_headway = 0.0
    count_time_headway = 0
    sum_acc_sq = 0.0
    count_acc_sq = 0
    sum_fuel_rate = 0.0
    count_fuel_rate = 0
    fuel_rate_clipped_count = 0
    metrics_skipped_reborn_or_oob = 0

    state = initial_state

    for i in range(1, iterations + 1):
        current_ttc = calculate_ttc(state, mode)
        current_ttc_category = get_ttc_category(current_ttc)
        ttc_counts[current_ttc_category] += 1

        if state_recorded:
            state_key = hash_state(state, mode)
            states_prob_record[state_key] = states_prob_record.get(state_key, 0) + 1

        if state is None or current_ttc <= 0:
            state = initial_state
        else:
            next_state_result, step_info = sample.sample_one_state(state, mode, dt=DT, return_info=True)
            next_ttc = calculate_ttc(next_state_result, mode)
            next_ttc_category = get_ttc_category(next_ttc)

            if mode in ("CF", "IDM") and step_info.get("metrics_eligible", True):
                thw = _compute_time_headway(state, mode)
                if thw is not None:
                    sum_time_headway += thw
                    count_time_headway += 1

                v_curr = float(state[0].velocity)
                # Use sampled AV action acceleration (bounded by policy), not state-diff jump.
                acc_fav = step_info.get("av_acc")
                if acc_fav is None:
                    v_next = float(next_state_result[0].velocity)
                    acc_fav = (v_next - v_curr) / DT
                acc_fav = float(acc_fav)
                sum_acc_sq += acc_fav * acc_fav
                count_acc_sq += 1

                fuel_rate = _vt_micro_fuel_rate(max(0.0, v_curr), acc_fav)
                if math.isfinite(fuel_rate):
                    if fuel_rate > VTM_FUEL_RATE_MAX:
                        fuel_rate = VTM_FUEL_RATE_MAX
                        fuel_rate_clipped_count += 1
                    sum_fuel_rate += fuel_rate
                    count_fuel_rate += 1
            elif mode in ("CF", "IDM"):
                metrics_skipped_reborn_or_oob += 1

            transition_counts[f"{current_ttc_category} to {next_ttc_category}"] += 1

            state = next_state_result

        if i % interval == 0:
            elapsed_time = round(time.time() - start_time, 2)
            start_time = time.time()

            log_entry = (i, interval_crash_count, elapsed_time)
            crash_log.append(log_entry)
            interval_crash_count = 0

            print(f"iterations: {i}, crashes in last {interval}: {log_entry}")

            with open(os.path.join(results_dir, "crash_log.txt"), "a") as f:
                f.write(f"{log_entry[0]}\t{log_entry[1]}\t{log_entry[2]}\n")

    additional_stats = {
        "avg_time_headway": (sum_time_headway / count_time_headway) if count_time_headway > 0 else None,
        "avg_acceleration_square": (sum_acc_sq / count_acc_sq) if count_acc_sq > 0 else None,
        "avg_fuel_consumption_rate_vtm": (sum_fuel_rate / count_fuel_rate) if count_fuel_rate > 0 else None,
        "fuel_consumption_rate_cap_vtm": VTM_FUEL_RATE_MAX,
        "fuel_rate_clipped_count": fuel_rate_clipped_count,
        "metrics_skipped_reborn_or_oob": metrics_skipped_reborn_or_oob,
        "time_headway_sample_count": count_time_headway,
        "acceleration_sample_count": count_acc_sq,
        "fuel_rate_sample_count": count_fuel_rate,
    }

    return ttc_counts, transition_counts, states_prob_record, critical_state, crash_log, additional_stats


def main():
    args = parse_args()
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    mode = args.mode
    initial_state = veh.generate_initial_state(mode)
    interval = args.interval
    if args.no_state_recorded:
        state_recorded = False
    elif args.state_recorded:
        state_recorded = True
    else:
        state_recorded = True
    iterations = args.iterations

    ttc_prob, transition_counts, states_prob_record, critical_state, crash_log, additional_stats = simulate_monte_carlo(
        initial_state, iterations, state_recorded, mode, interval=interval, results_dir=results_dir
    )

    ttc_prob_filename = os.path.join(results_dir, f"ttc_prob_benchmark_data.json")
    save_to_json(ttc_prob, ttc_prob_filename)

    transition_data = {}
    for from_ttc in ttc_ranges:
        total_transitions_from_ttc = sum(
            v for k, v in transition_counts.items() if k.startswith(f"{from_ttc} to")
        )

        transition_data[from_ttc] = {
            k.split(" to ")[1]: {
                "count": v,
                "probability": (v / total_transitions_from_ttc) if total_transitions_from_ttc > 0 else 0,
            }
            for k, v in transition_counts.items()
            if k.startswith(f"{from_ttc} to")
        }

    save_to_json(critical_state, os.path.join(results_dir, "critical_states.json"))
    transition_data_filename = os.path.join(results_dir, f"transition_data.json")
    save_to_json(transition_data, transition_data_filename)
    save_to_json(additional_stats, os.path.join(results_dir, "additional_stats.json"))

    if state_recorded:
        save_to_json(states_prob_record, os.path.join(results_dir, "states_record.json"))


if __name__ == "__main__":
    main()
