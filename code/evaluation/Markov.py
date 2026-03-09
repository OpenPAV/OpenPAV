from datetime import datetime
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import data_analysis as pt
from utils import *
import sample_scenarios as dm
import sys
import os
import gc
import argparse
import time

# Get the project root directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
results_dir = os.path.join(base_dir, f'results/Markov/1216')

os.environ["QT_QPA_FONTDIR"] = "path_to_your_fonts"
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def _append_markov_log(message: str) -> None:
    log_path = os.path.join(results_dir, "runtime_log.txt")
    with open(log_path, "a") as f:
        f.write(message + "\n")


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _enforce_crash_rebirth_row(transition_counts):
    """
    Keep Markov transition output consistent with the crash-rebirth modeling rule:
    crash TTC bucket (-inf, 0) transitions to the safest bucket (8, inf) with prob 1.
    """
    crash_from = f"{-float('inf')}-0"
    safe_to = (8, float("inf"))
    transition_counts[crash_from] = {safe_to: 1.0}
    return transition_counts



def iterate_ttc_analysis(initial_state_weights_file, num_samples, mode="CF", stability_threshold=0.001):
    """Iterate over TTC ranges and accumulate transitions between TTC categories."""
    # Initialize the structure to accumulate transition counts across all iterations
    accumulated_transition_counts = {f"{range_[0]}-{range_[1]}": {} for range_ in reversed_ttc_ranges}
    dm.state_weights={}
    total_iteration=0
    global_start = time.time()
    _append_markov_log(f"[{_ts()}] Start Markov iterate_ttc_analysis mode={mode}, num_samples={num_samples}, stability={stability_threshold}")
    
    for iteration in range(len(reversed_ttc_ranges)):
        gc.collect()
        ttc_range = reversed_ttc_ranges[iteration]
        print(f"Starting analysis for TTC range {ttc_range}")
        stage_start = time.time()
        _append_markov_log(f"[{_ts()}] Start TTC block {ttc_range} (index={iteration})")
        iteration_count = 0
        sample_offset = 0
        stable = 0
        previous_means = {}
        crash_hits_total = 0
        total_transition_counts = {}  # accumulate transition_counts
        while True:
            loop_start = time.time()
            if iteration == 0:
                # First block: process all benchmark states once, then move on.
                if iteration_count > 0:
                    _append_markov_log(f"[{_ts()}] Block {ttc_range} first-block one-pass finished; stop stage.")
                    break
                sampled_states_with_prob = dm.initialize_from_file(initial_state_weights_file, ttc_range, mode)
                exhausted = True
            else:
                sampled_states_with_prob, sample_offset, exhausted = dm.sample_states_sequential(
                    ttc_range, num_samples, start_idx=sample_offset
                )
            if not sampled_states_with_prob:
                _append_markov_log(f"[{_ts()}] Block {ttc_range} no more states to traverse; stop stage.")
                break

            transition_counts, crash_hits = dm.analyze_sampled_states(sampled_states_with_prob, ttc_range, mode)
            crash_hits_total += crash_hits
            iteration_count += 1
            print("complete sampling ",iteration_count,f" at {ttc_range}")

            # Accumulate current transition counts into the total_transition_counts
            for from_category, to_categories in transition_counts.items():
                if from_category not in total_transition_counts:
                    total_transition_counts[from_category] = {}
                for to_category, count in to_categories.items():
                    if to_category in total_transition_counts[from_category]:
                        total_transition_counts[from_category][to_category] += count
                    else:
                        total_transition_counts[from_category][to_category] = count

            total_iteration += 1
            if iteration < len(reversed_ttc_ranges) - 1:
                next_range = reversed_ttc_ranges[iteration + 1]
                normalized_transition = normalize_counts(total_transition_counts)
                current_means = normalized_transition

                if previous_means:
                    max_change = get_max_change_rate(previous_means, current_means)
                    if ((0 <= max_change < stability_threshold) and iteration_count > ITE_MIN_CONUT) or iteration_count > ITE_MAX_CONUT:
                        stable += 1
                    else:
                        stable = 0

                    if stable > STABLE_CONUT:
                        if next_range[0] == -float("inf") and next_range[1] == 0:
                            ready_for_next = crash_hits_total >= num_samples
                        else:
                            next_bucket_count = len(
                                {
                                    state: prob
                                    for state, prob in dm.state_weights.items()
                                    if next_range[0] <= state[-1] < next_range[1]
                                }
                            )
                            ready_for_next = next_bucket_count >= num_samples
                        if ready_for_next:
                            _append_markov_log(
                                f"[{_ts()}] Block {ttc_range} reached stability with enough next-bucket states; stop stage."
                            )
                            break
                previous_means = current_means

            loop_elapsed = time.time() - loop_start
            stage_elapsed = time.time() - stage_start
            _append_markov_log(
                f"[{_ts()}] Block {ttc_range} loop={iteration_count} "
                f"loop_elapsed={loop_elapsed:.2f}s stage_elapsed={stage_elapsed:.2f}s stable={stable} crash_hits_total={crash_hits_total}"
            )
            if exhausted:
                _append_markov_log(f"[{_ts()}] Block {ttc_range} traversed all sampled states; stop stage.")
                break

        # 调用规范化和累积函数
        normalized_transition=normalize_counts(total_transition_counts)
        accumulated_transition_counts=accumulate_counts(normalized_transition, accumulated_transition_counts)
        accumulated_transition_counts = _enforce_crash_rebirth_row(accumulated_transition_counts)
        save_to_json(accumulated_transition_counts, os.path.join(results_dir, 'accumulated_transition_counts.json'))
        _append_markov_log(f"[{_ts()}] Done TTC block {ttc_range} elapsed={time.time() - stage_start:.2f}s")

    # Save accumulated transition counts for all iterations
    accumulated_transition_counts = _enforce_crash_rebirth_row(accumulated_transition_counts)
    save_to_json(accumulated_transition_counts, os.path.join(results_dir, 'accumulated_transition_counts.json'))
    _append_markov_log(f"[{_ts()}] Done Markov iterate_ttc_analysis total_elapsed={time.time() - global_start:.2f}s")
    return iteration,total_iteration



def ite_test():
    thresholds = [0.3,0.02,0.002]
    repeat_per_thr = 5
    results = {}

    summary_dir = os.path.join(base_dir, 'results/Markov/1216/summary')
    os.makedirs(summary_dir, exist_ok=True)
    summary_file = os.path.join(summary_dir, 'total_iterations.txt')


    for thr in thresholds:
        results[thr] = []
        for repeat in range(repeat_per_thr):
            print(f"xx: thr_{thr}_rep_{repeat+1}")
            global results_dir
            results_dir = os.path.join(base_dir, f'results/Markov/1216/thr_{thr}_rep_{repeat+1}')
            os.makedirs(results_dir, exist_ok=True)

            iteration,total_iteration = iterate_ttc_analysis(
                initial_state_weights_file=os.path.join(base_dir, 'states_record_LC.jsonl'),
                num_samples=Num_samples_per_ite,
                mode="LC",
                stability_threshold=thr
            )

            line = f'{thr},{repeat+1},{iteration},{total_iteration}\n'
            with open(summary_file, 'a') as f:
                f.write(line)

    print('Done. Results saved to', summary_file)

def single_test():
    record_crash=False
    results_dir = os.path.join(base_dir, f'results/Markov/1216')#f'Debug')#
    os.makedirs(results_dir, exist_ok=True)

    # 运行实验
    total_iteration=iterate_ttc_analysis(
        initial_state_weights_file=os.path.join(base_dir, 'results/benchmark/1216/states_record.json'),
        num_samples=Num_samples_per_ite,
        mode="CF",
        stability_threshold= Stability_threshold
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default=results_dir, help="Output directory for Markov results")
    parser.add_argument("--initial_state_weights_file", type=str, default=os.path.join(base_dir, 'results/benchmark/1216/states_record.json'))
    parser.add_argument("--num_samples", type=int, default=Num_samples_per_ite)
    parser.add_argument("--mode", type=str, default="CF")
    parser.add_argument("--stability_threshold", type=float, default=Stability_threshold)
    args = parser.parse_args()

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    iterate_ttc_analysis(
        initial_state_weights_file=args.initial_state_weights_file,
        num_samples=args.num_samples,
        mode=args.mode,
        stability_threshold=args.stability_threshold,
    )
