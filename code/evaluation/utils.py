import json
import numpy as np
import csv
import copy

# Parameters
iteration_counts = {}
DT = 0.2  # [s]

# Sequential algorithm
# ttc_interval = 0.5
# max_ttc_value = 8.0
granularity = 0.1  # granularity for states
hash_granularity = 0.1  # Hash granularity for states
action_granularity = 0.1  # granularity when sampling actions

min_speed = 0
max_speed = 50
max_space = 80

# Hash-grid size (counts of bins), not physical values.
num_speed_bins = int(round((max_speed - min_speed) / hash_granularity)) + 1
num_space_bins = int(round(max_space / hash_granularity)) + 1
state_ranges = [3, num_speed_bins, num_space_bins]  # lane, velocity, space
car_length = 5.0

# BV driving model parameters
# NDD Vehicle IDM parameters
NDD_COMFORT_ACC_MAX = 2  # [m/s2]
NDD_COMFORT_ACC_MIN = 4.0  # [m/s2]
NDD_DISTANCE_WANTED = 5.0  # [m]
NDD_TIME_WANTED = 1.5  # [s]
NDD_DESIRED_VELOCITY = 35  # [m/s]
NDD_DELTA = 4.0  # []
# NDD Vehicle MOBIL parameters
NDD_POLITENESS = 0.1  # in [0, 1]  0.5
NDD_LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]  0.2
NDD_LANE_CHANGE_MAX_BRAKING_IMPOSED = 3.0  # [m/s2]  2
NDD_MOBIL_LC_prob = 1e-2
# CAV Lateral policy parameters (MOBIL)
CAV_POLITENESS = 0.  # in [0, 1]
CAV_LANE_CHANGE_MIN_ACC_GAIN = 0.1  # [m/s2]
CAV_LANE_CHANGE_MAX_BRAKING_IMPOSED = 4.0  # [m/s2]
CAV_MOBIL_LC_prob = 1e-6
Surrogate_LANE_CHANGE_MAX_BRAKING_IMPOSED = 4.0  # [m/s2]
CAV_LANE_CHANGE_DELAY = 1.0  # [s]

# CAV Longitudinal CAV policy parameters (henry)
CAV_COMFORT_ACC_MAX = 1.5  # maximum comfortable acceleration for CAV IDM model
CAV_COMFORT_ACC_MIN = 2  # minimum comfortable acceleration for CAV IDM model
CAV_DESIRED_VELOCITY = 35  # [m/s]
CAV_TIME_WANTED = 1.0  # [s]
CAV_DISTANCE_WANTED = 2.0  # [m]
CAV_DELTA = 4.0  # acceleration exponent []

# Distribution
sigma_acc = 1.5  # standard deviation of acceleration noise
a_min = -2
a_max = 4
distribution_ignore_threshold = 1e-2

# henry nde
acc_low, acc_high = -4, 2
acc_resolution = 0.2
num_acc = int(1 + ((acc_high - acc_low) / acc_resolution))
acc_list = list(np.linspace(acc_low, acc_high, num=num_acc))

# Sample process
Num_samples_per_ite = 10000
Stability_threshold = 0.1
ITE_CONUT_CANNOT_FIND_NEXT = 10
ITE_MIN_CONUT = 3
ITE_MAX_CONUT = 100
STABLE_CONUT = 3
MAX_Checked_actions = 40  # max checked actions each time


# if 1:#debug
#     DT = 2#0.85  # [s]
#     ttc_interval = 2
#     max_ttc_value = 8.0
#     granularity = 0.1  # granularity for states
#     Num_samples_per_ite=100
#     Stability_threshold=1
#     ITE_CONUT_CANNOT_FIND_NEXT=5
#     ITE_MIN_CONUT=1
#     ITE_MAX_CONUT=5


def threshold_and_normalize(lst, threshold):
    arr = np.array(lst)
    arr[arr < threshold] = 0  # 小于阈值的值设为 0

    if np.sum(arr) > 0:  # 避免全为 0 时除以 0
        arr = (arr - np.min(arr)) / np.sum(arr)

    return arr


# Util functions
def generate_ttc_ranges(interval, max_value, include_inf=True):
    ttc_ranges = []
    ttc_ranges.append((-float('inf'), 0))
    current = 0.0
    while current < max_value:
        next_value = current + interval
        if next_value > max_value:
            next_value = max_value
        ttc_ranges.append((current, next_value))
        current = next_value
    if include_inf:
        ttc_ranges.append((max_value, float('inf')))
    return ttc_ranges


# Util functions
def generate_ttc_ranges_ln():
    ttc_ranges = []
    # Negative to 0
    ttc_ranges.append((-float('inf'), 0))
    # Log-style positive ranges
    boundaries = [0, 0.25, 0.5, 1.5, 3, 5, 8, float('inf')]
    for i in range(len(boundaries) - 1):
        ttc_ranges.append((boundaries[i], boundaries[i + 1]))
    return ttc_ranges


# create some fixed-parameters
ttc_ranges = generate_ttc_ranges_ln()
# ttc_ranges = generate_ttc_ranges_ln(ttc_interval, max_ttc_value)
reversed_ttc_ranges = list(reversed(ttc_ranges))


def load_ttc_ranges_from_file(filename):
    ttc_ranges = []
    with open(filename, 'r') as f:
        data = json.load(f)
        for key in data.keys():
            if ',' in key:
                lower, upper = key.strip("()").split(',')
                ttc_ranges.append((float(lower), float(upper)))
    return ttc_ranges


def get_ttc_category(ttc):
    if ttc < 0:
        return -float('inf'), 0
    for lower, upper in ttc_ranges:
        if lower <= ttc < upper:
            return lower, upper
    return ttc_ranges[-1][0], float('inf')


def hash_state(state, mode):
    if mode in ("CF", "IDM"):
        FAV, LBV = state
        if LBV.space < 0:
            return -1
        v_lead_index = int(round((LBV.velocity - min_speed) / hash_granularity))
        v_index = int(round((FAV.velocity - min_speed) / hash_granularity))
        s_index = int(round(LBV.space / hash_granularity))

        # Clamp indices to configured state domain for robust hashing.
        v_lead_index = max(0, min(v_lead_index, state_ranges[1] - 1))
        v_index = max(0, min(v_index, state_ranges[1] - 1))
        s_index = max(0, min(s_index, state_ranges[2] - 1))

        hash_value = v_lead_index * state_ranges[1] * state_ranges[2] + v_index * state_ranges[2] + s_index
        return hash_value




def unhash_state(hash_value, mode="CF"):
    # Local import to avoid circular dependency at module import time
    from vehicles import Vehicle

    if mode in ("CF", "IDM"):
        if int(hash_value) == -1:
            return Vehicle(0, 0, 0), Vehicle(0, -1, 0)
        s_index = hash_value % state_ranges[2]
        v_index = (hash_value // state_ranges[2]) % state_ranges[1]
        v_lead_index = hash_value // (state_ranges[1] * state_ranges[2])

        fav_velocity = min_speed + v_index * hash_granularity
        lead_velocity = min_speed + v_lead_index * hash_granularity
        space = s_index * hash_granularity
        FAV = Vehicle(fav_velocity, 0, 0)
        LBV = Vehicle(lead_velocity, space, 0)
        return FAV, LBV



def save_to_json(data, filename):
    def convert_keys_to_str(d):
        """Recursively convert dictionary keys to strings if they are tuples."""
        if isinstance(d, dict):
            return {str(k): convert_keys_to_str(v) for k, v in d.items()}
        return d

    data = convert_keys_to_str(data)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def save_to_csv(means, filename):
    """Append the current means to a CSV file."""
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        # 将 means 字典转化为列表，并写入文件
        writer.writerow([means['v'], means['v_lead'], means['s']])


def load_from_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def cal_state_para_means(state_weights_next):
    """
    计算 AV 和每辆 BV 的空间 (space) 和速度 (velocity) 均值，根据 mode 进行不同处理。

    :param state_weights_next: 字典，键是 (FAV, BVs) 的元组，值是对应权重
    :param mode: 字符串，决定计算哪类车辆的参数
    :return: 包含 AV 和每辆 BV 速度 & 空间均值的字典
    """
    total_weight = sum(state_weights_next.values())

    if total_weight == 0:
        raise ValueError("Total weight is zero, cannot divide by zero.")

    # 初始化 AV 变量
    av_velocities = []
    # 初始化 BV 字典
    bv_velocities = {}  # {BV_id: [weighted velocity]}
    bv_spaces = {}  # {BV_id: [weighted space]}

    for (state, ttc), prob in state_weights_next.items():
        if np.isnan(prob):
            continue  # 跳过无效数据

        AV = state[0]
        BVs = state[0:]

        # 处理 AV
        if not np.isnan(AV.velocity) and not np.isnan(AV.space):
            av_velocities.append(AV.velocity * prob)

        # 处理每一辆 BV
        for BV in BVs:
            if not np.isnan(BV.velocity) and not np.isnan(BV.space):
                if BV not in bv_velocities:
                    bv_velocities[BV] = []
                    bv_spaces[BV] = []

                bv_velocities[BV].append(BV.velocity * prob)
                bv_spaces[BV].append(BV.space * prob)

    # 计算均值
    current_means = {
        "AV_v": np.sum(av_velocities) / total_weight if av_velocities else float('nan')
    }

    # 计算每辆 BV 的均值，并展开成独立的字典键
    for i, BV in enumerate(bv_velocities, start=1):
        current_means[f"BV_v_{i}"] = np.sum(bv_velocities[BV]) / total_weight if bv_velocities[BV] else float(
            'nan')
        current_means[f"BV_s_{i}"] = np.sum(bv_spaces[BV]) / total_weight if bv_spaces[BV] else float('nan')

    return current_means




def normalize_counts(total_transition_counts):
    """Normalize the total transition counts and accumulate into accumulated_transition_counts."""
    normalized_transition = copy.deepcopy(total_transition_counts)
    # Normalize total_transition_counts
    total_prob = sum(sum(to_categories.values()) for to_categories in normalized_transition.values())
    if total_prob > 0:
        for from_category in normalized_transition:
            for to_category in normalized_transition[from_category]:
                normalized_transition[from_category][to_category] /= total_prob
    return normalized_transition


def accumulate_counts(normalized_transition, accumulated_transition_counts):
    # Accumulate total transition counts
    for from_category, to_categories in normalized_transition.items():
        if from_category not in accumulated_transition_counts:
            accumulated_transition_counts[from_category] = {}
        for to_category, count in to_categories.items():
            if to_category in accumulated_transition_counts[from_category]:
                accumulated_transition_counts[from_category][to_category] += count
            else:
                accumulated_transition_counts[from_category][to_category] = count
    return accumulated_transition_counts


def get_max_change_rate(prev_normalized, normalized):
    max_rate = -1

    for from_cat in normalized:
        for to_cat in normalized[from_cat]:
            prev_val = prev_normalized.get(from_cat, {}).get(to_cat, 0)
            curr_val = normalized[from_cat][to_cat]

            if prev_val == 0:
                change_rate = float('inf') if curr_val != 0 else 0
            else:
                change_rate = abs((curr_val - prev_val) / prev_val)

            if change_rate > max_rate:
                max_rate = change_rate

    return max_rate


def cal_state_para_means_modified(state_weights_next):
    """
    计算状态转换参数均值，包括 AV 速度均值和 BV 接近指数均值。
    
    :param state_weights_next: 字典，键是 (FAV, BVs) 的元组，值是对应权重
    :return: 包含 AV 速度均值和 BV 接近指数均值的字典
    """
    total_weight = sum(state_weights_next.values())

    if total_weight == 0:
        print("Total weight is zero, cannot divide by zero.")
        return None

    # 初始化 AV 速度
    av_velocities = []

    # 初始化接近指数存储
    proximity_indices = {i: [] for i in range(1, 7)}  # 1-6 共六个接近指数位置

    for (state, ttc), prob in state_weights_next.items():
        if np.isnan(prob):
            continue  # 跳过无效数据

        AV = state[0]
        BVs = state[1:][0]  # 剔除 AV，剩下的都是 BV

        # 处理 AV 速度
        if not np.isnan(AV.velocity):
            av_velocities.append(AV.velocity * prob)

        # 分类 BV 到不同的车道
        lane_groups = {0: [], 1: [], 2: []}  # {lane: [(BV, distance, proximity_index)]}

        for BV in BVs:
            if np.isnan(BV.velocity) or np.isnan(BV.space) or np.isnan(BV.lane_id):
                continue  # 跳过无效数据

            rel_velocity = BV.velocity - AV.velocity  # 相对速度
            if rel_velocity == 0:
                continue  # 避免除零错误

            proximity_index = BV.space / rel_velocity  # 计算接近指数
            lane_groups[BV.lane_id].append((BV, BV.space, proximity_index))

        # 选取每个车道上空间最接近 AV 的正负值各 1 个
        for lane, bv_list in lane_groups.items():
            if not bv_list:
                continue  # 该车道无 BV

            # 分成正空间和负空间
            positive_space = [x for x in bv_list if x[1] > 0]  # 前方车辆
            negative_space = [x for x in bv_list if x[1] < 0]  # 后方车辆

            # 选取最接近 AV 的正负各 1 个
            closest_positive = min(positive_space, key=lambda x: abs(x[1]), default=None)
            closest_negative = min(negative_space, key=lambda x: abs(x[1]), default=None)

            if lane == AV.lane_id:  # 本车道
                if closest_positive:
                    proximity_indices[1].append(closest_positive[2] * prob)
                if closest_negative:
                    proximity_indices[2].append(closest_negative[2] * prob)

            elif lane == AV.lane_id - 1:  # 左侧车道
                if closest_positive:
                    proximity_indices[3].append(closest_positive[2] * prob)
                if closest_negative:
                    proximity_indices[4].append(closest_negative[2] * prob)

            elif lane == AV.lane_id + 1:  # 右侧车道
                if closest_positive:
                    proximity_indices[5].append(closest_positive[2] * prob)
                if closest_negative:
                    proximity_indices[6].append(closest_negative[2] * prob)

    # 计算均值
    current_means = {
        "AV_v": np.sum(av_velocities) / total_weight if av_velocities else float('nan')
    }

    # 计算接近指数均值
    for i in range(1, 7):
        current_means[f"pi_{i}"] = (np.nanmean(proximity_indices[i]) if proximity_indices[i] else float('nan')
                                    )

    return current_means


# Function to calculate relative kinetic energy for collision severity
def calculate_collision_severity(v_leading, v_following, mass=1000):
    v_rel = abs(v_following - v_leading)
    kinetic_energy = 0.5 * mass * v_rel ** 2
    return kinetic_energy


def calculate_ttc(state, mode):
    """ 计算 AV 的 TTC，根据 CF 或 LC 模式 """

    def compute_ttc(v_lead, v, s):
        """ 计算 TTC (Time-to-Collision) """
        # Explicit crash handling: ensure crash states are mapped to negative TTC.
        if s <= 0:
            return -1.0
        if v_lead > v:
            return float('inf')
        elif v == v_lead:
            return float('inf')
        else:
            return round((s - car_length) / (v - v_lead), 1)

    AV = state[0]  # 获取 AV

    if mode in ("CF", "IDM"):
        LBV = state[1]  # 最近前方背景车辆
        return compute_ttc(LBV.velocity, AV.velocity, LBV.space)



def calculate_weighted_collision_severity_distribution(file_path, benchmark=False):
    with open(file_path, 'r') as f:
        data = json.load(f)

    severities = []
    weights = []
    if benchmark:
        for key, state in data.items():

            if calculate_ttc(state) < 0.1:
                v_leading, v_following, distance = state
                severity = calculate_collision_severity(v_leading, v_following)
                severities.append(severity)
                weights.append(1)

        return severities, weights
    # Filter states with TTC < 0.5 and calculate weighted severity
    for key, probability in data.items():
        v_leading, v_following, distance, ttc = eval(key)

        if ttc < 0.25:
            severity = calculate_collision_severity(v_leading, v_following)
            severities.append(severity)
            weights.append(probability)

    return severities, weights
