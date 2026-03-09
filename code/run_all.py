import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime



def parse_args():
    parser = argparse.ArgumentParser(description="Run behavior modeling + MonteCarlo + Markov pipeline.")
    parser.add_argument(
        "--data_csv",
        type=str,
        default=None,
        help="CSV file in data folder. Required for learned-model runs; optional for IDM benchmark.",
    )
    parser.add_argument(
        "--fav_id",
        type=int,
        default=None,
        help="Target FAV ID. Required for learned-model runs; optional for IDM benchmark.",
    )
    parser.add_argument("--seq_len", type=int, default=1, help="Sequence length for LSTM")
    parser.add_argument("--train_epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train/val split ratio")
    parser.add_argument("--mc_iterations", type=int, default=1000000, help="MonteCarlo iterations")
    parser.add_argument("--mc_interval", type=int, default=10000, help="MonteCarlo logging interval")
    parser.add_argument("--markov_samples", type=int, default=500, help="Samples per Markov iteration")
    parser.add_argument("--markov_stability", type=float, default=0.05, help="Markov stability threshold")
    parser.add_argument("--mode", type=str, default="CF", choices=["CF", "IDM"], help="Simulation mode")
    parser.add_argument(
        "--benchmark_model",
        type=str,
        default="learned",
        choices=["learned", "idm"],
        help="Benchmark source for AV dynamics: learned model or stochastic IDM baseline.",
    )
    parser.add_argument("--results_root", type=str, default="results", help="Root results directory")
    parser.add_argument("--skip_train", action="store_true", help="Skip training step")
    parser.add_argument("--skip_mc", action="store_true", help="Skip MonteCarlo step")
    parser.add_argument("--skip_markov", action="store_true", help="Skip Markov step")
    return parser.parse_args()


def run_cmd(cmd, env=None):
    result = subprocess.run(cmd, env=env, check=True)
    return result.returncode


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _elapsed_str(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _module_log(label: str, start_ts: float = None) -> float:
    if start_ts is None:
        print(f"[{_now_str()}] Start {label}")
        return datetime.now().timestamp()
    elapsed = datetime.now().timestamp() - start_ts
    print(f"[{_now_str()}] Done {label} (elapsed {_elapsed_str(elapsed)})")
    return elapsed


def _has_training_outputs(bm_dir: Path) -> bool:
    return (bm_dir / "best.pth").exists() and (bm_dir / "calibration.json").exists()


def _has_mc_outputs(mc_dir: Path) -> bool:
    return (mc_dir / "states_record.json").exists()


def _has_markov_outputs(mk_dir: Path) -> bool:
    return (mk_dir / "accumulated_transition_counts.json").exists()


def _has_steady_outputs(mk_dir: Path) -> bool:
    return (mk_dir / "steady_state_result.json").exists()


def main():
    args = parse_args()
    selected_mode = args.mode
    skip_train = args.skip_train
    is_idm_benchmark = (args.benchmark_model == "idm") or (args.mode == "IDM")

    if is_idm_benchmark:
        selected_mode = "IDM"
        skip_train = True
        run_root = Path(args.results_root) / "IDM"
        data_csv = None
    else:
        if args.fav_id is None:
            raise ValueError("--fav_id is required for learned-model runs.")
        if not args.data_csv:
            raise ValueError("--data_csv is required for learned-model runs.")
        data_csv = Path(args.data_csv).resolve()
        if not data_csv.exists():
            raise FileNotFoundError(f"CSV not found: {data_csv}")
        data_name = data_csv.parent.name
        run_root = Path(args.results_root) / f"{data_name}_FAV{args.fav_id}"
    bm_dir = run_root / "behavior_modeling"
    mc_dir = run_root / "benchmark"
    mk_dir = run_root / "Markov"

    bm_dir.mkdir(parents=True, exist_ok=True)
    mc_dir.mkdir(parents=True, exist_ok=True)
    mk_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    if not skip_train:
        if _has_training_outputs(bm_dir):
            print(f"[{_now_str()}] Skip training: outputs already exist in {bm_dir}")
        else:
            t0 = _module_log("behavior_modeling/train")
            train_cmd = [
                python,
                "behavior_modeling/train.py",
                "--data_dir",
                str(data_csv.parent),
                "--data_csv",
                str(data_csv),
                "--fav_id",
                str(args.fav_id),
                "--seq_len",
                str(args.seq_len),
                "--num_epochs",
                str(args.train_epochs),
                "--batch_size",
                str(args.train_batch_size),
                "--train_ratio",
                str(args.train_ratio),
                "--result_dir",
                str(bm_dir),
            ]
            run_cmd(train_cmd)
            _module_log("behavior_modeling/train", start_ts=t0)

    env = os.environ.copy()
    env["CF_MODEL_DIR"] = str(bm_dir)

    if not args.skip_mc:
        if _has_mc_outputs(mc_dir):
            print(f"[{_now_str()}] Skip MonteCarlo: outputs already exist in {mc_dir}")
        else:
            t0 = _module_log("evaluation/MonteCarlo")
            mc_cmd = [
                python,
                "evaluation/MonteCarlo.py",
                "--results_dir",
                str(mc_dir),
                "--iterations",
                str(args.mc_iterations),
                "--interval",
                str(args.mc_interval),
                "--mode",
                selected_mode,
                "--state_recorded",
            ]
            run_cmd(mc_cmd, env=env)
            _module_log("evaluation/MonteCarlo", start_ts=t0)

    if not args.skip_markov:
        states_record = mc_dir / "states_record.json"
        if not states_record.exists():
            raise FileNotFoundError(f"states_record.json not found: {states_record}")
        if _has_markov_outputs(mk_dir):
            print(f"[{_now_str()}] Skip Markov: outputs already exist in {mk_dir}")
        else:
            t0 = _module_log("evaluation/Markov")
            markov_cmd = [
                python,
                "evaluation/Markov.py",
                "--results_dir",
                str(mk_dir),
                "--initial_state_weights_file",
                str(states_record),
                "--mode",
                selected_mode,
            ]
            if args.markov_samples is not None:
                markov_cmd += ["--num_samples", str(args.markov_samples)]
            if args.markov_stability is not None:
                markov_cmd += ["--stability_threshold", str(args.markov_stability)]
            run_cmd(markov_cmd, env=env)
            _module_log("evaluation/Markov", start_ts=t0)

        if _has_steady_outputs(mk_dir):
            print(f"[{_now_str()}] Skip steady-state: outputs already exist in {mk_dir}")
        else:
            t0 = _module_log("evaluation/markov_steady_state")
            steady_cmd = [
                python,
                "evaluation/markov_steady_state.py",
                str(mk_dir / "accumulated_transition_counts.json"),
            ]
            run_cmd(steady_cmd)
            _module_log("evaluation/markov_steady_state", start_ts=t0)


if __name__ == "__main__":
    main()
