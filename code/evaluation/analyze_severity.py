import os
import sys
import pandas as pd

# Get the project root directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
results_dir = os.path.join(base_dir, 'Results/Debug')

from Markov import utils as bf
from Markov import data_analysis as pt


def proposed():
    severities, weights = bf.calculate_weighted_collision_severity_distribution(
        os.path.join(results_dir, f'States_distribution/State_Distribution_0.25_0.5.json'))

    pt.plot_collision_severity_distribution(severities, weights, "Collision Severity Distribution",
                                            output_dir="./plots", issave=True)


def benchmark(filename):
    severities, weights = bf.calculate_weighted_collision_severity_distribution(filename, True)
    # 假设 severities 和 weights 是两个数组
    data = {
        'Severity': severities,
        'Weight': weights
    }

    # 创建 DataFrame 并保存为 CSV
    df = pd.DataFrame(data)
    df.to_csv('severities_weights.csv', index=False)

    pt.plot_collision_severity_distribution(severities, weights, "Collision Severity Distribution_Benchmark",
                                            output_dir="./plots", issave=True)


# proposed()
benchmark("Results\Severatiy\c_state.json")
