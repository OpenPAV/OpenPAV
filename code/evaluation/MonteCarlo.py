import os
import sys
import time
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from utils import *  # noqa: F401,F403
import sample_scenarios as sample
import vehicles as veh


print("Current Working Directory:", os.getcwd())
DEFAULT_RESULTS_DIR = os.path.join(BASE_DIR, "results", "benchmark")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Output directory for MonteCarlo results")
    parser.add_argument("--iterations", type=int, default=100000, help="Total MonteCarlo iterations")
    parser.add_argument("--interval", type=int, default=10000, help="Logging interval")
    parser.add_argument("--state_recorded", action="store_true", help="Record state distribution")
    parser.add_argument("--no_state_recorded", action="store_true", help="Disable state recording")
    parser.add_argument("--mode", type=str, default="CF", help="Simulation mode")
    return parser.parse_args()


def simulate_monte_carlo(initial_state, iterations, state_recorded, mode, interval, results_dir):
    start_time = time.time()

    critical_state = {}
    ttc_counts = {ttc_range: 0 for ttc_range in ttc_ranges}
    transition_counts = {f"{from_ttc} to {to_ttc}": 0 for from_ttc in ttc_counts for to_ttc in ttc_counts}
    states_prob_record = {}

    crash_log = []
    interval_crash_count = 0

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
            next_state_result = sample.sample_one_state(state, mode, dt=DT)
            next_ttc = calculate_ttc(next_state_result, mode)
            next_ttc_category = get_ttc_category(next_ttc)

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

    return ttc_counts, transition_counts, states_prob_record, critical_state, crash_log


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

    ttc_prob, transition_counts, states_prob_record, critical_state, crash_log = simulate_monte_carlo(
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

    if state_recorded:
        save_to_json(states_prob_record, os.path.join(results_dir, "states_record.json"))


if __name__ == "__main__":
    main()
