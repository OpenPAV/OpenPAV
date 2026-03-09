import sparse
from bidict import bidict
import bisect
import scipy
from utils import *
import vehicles as veh

import os
# import data
print(os.path.dirname(os.path.abspath(__file__)))
ndd_data_path = "./data/NDE/NDD_DATA"

# initial state
speed_CDF = np.load(ndd_data_path + "/Initialization/speed_CDF.npy")
presum_list_forward = np.load(
    ndd_data_path + "/Initialization/Optimized_presum_list_forward.npy")  # choose prev vehicle according to range and range rate, according to the velocity of the following vehicle
random_veh_pos_buffer_start, random_veh_pos_buffer_end = 0, 50  # the first vehicle in the traffic flow will be generated between random_veh_pos_buffer_start and random_veh_pos_buffer_end
ff_dis = 120  # free flow minimum distance
CF_percent = 0.6823141  # how many vehicles are in car following mode, otherwise in free flow mode

# driving behaviors
# CF
CF_pdf_array = np.load(
    ndd_data_path + "/CF/Optimized_CF_pdf_array.npy")  # NDD distribution to generate the longitudinal decision (acceleration) under car following scenarios
FF_pdf_array = np.load(
    ndd_data_path + "/FF/Optimized_FF_pdf_array.npy")  # NDD distribution to generate longitudinal decisions (acceleration) under free flow scenarios
# LC
OL_pdf = sparse.load_npz(ndd_data_path + "/LC/10106_10620_OL_pdf_smoothed_soft_p8_1_tmp.npz").todense()
SLC_pdf = sparse.load_npz(ndd_data_path + "/LC/10119_10630_SLC_pdf_smoothed_soft_p8_1_tmp.npz").todense()
DLC_pdf = sparse.load_npz(ndd_data_path + "/LC/10119_10630_DLC_pdf_smoothed_p8_1_tmp.npz").todense()
CI_pdf = sparse.load_npz(ndd_data_path + "/LC/10119_10630_CI_pdf_smoothed_p8_1_tmp.npz").todense()

# NDD distribution bound
v_low, v_high, r_low, r_high, rr_low, rr_high, acc_low, acc_high = 20, 40, 0, 115, -10, 8, -4, 2
v_resolution, r_resolution, rr_resolution, acc_resolution = 1, 1, 1, 0.2
num_v, num_r, num_rr, num_acc = int(1 + ((v_high - v_low) / v_resolution)), \
    int(1 + ((r_high - r_low) / r_resolution)), \
    int(1 + ((rr_high - rr_low) / rr_resolution)), \
    int(1 + ((acc_high - acc_low) / acc_resolution))
speed_list, r_list, rr_list, acc_list = list(np.linspace(v_low, v_high, num=num_v)), \
    list(np.linspace(r_low, r_high, num=num_r)), \
    list(np.linspace(rr_low, rr_high, num=num_rr)), \
    list(np.linspace(acc_low, acc_high, num=num_acc))
r_to_idx_dic, rr_to_idx_dic, v_to_idx_dic, acc_to_idx_dic = bidict(), bidict(), bidict(), bidict()
for i in range(num_r): r_to_idx_dic[list(np.linspace(r_low, r_high, num=num_r))[i]] = i
for j in range(num_rr): rr_to_idx_dic[list(np.linspace(rr_low, rr_high, num=num_rr))[j]] = j
for k in range(num_v): v_to_idx_dic[list(np.linspace(v_low, v_high, num=num_v))[k]] = k
for m in range(num_acc): acc_to_idx_dic[list(np.linspace(acc_low, acc_high, num=num_acc))[m]] = m

# NDD distribution for One lead lateral decision
one_lead_v_to_idx_dic, one_lead_r_to_idx_dic, one_lead_rr_to_idx_dic = bidict(), bidict(), bidict()
one_lead_r_low, one_lead_r_high, one_lead_rr_low, one_lead_rr_high, one_lead_v_low, one_lead_v_high = 0, 115, -10, 8, 20, 40
one_lead_r_step, one_lead_rr_step, one_lead_v_step = 1, 1, 1
one_lead_speed_list, one_lead_r_list, one_lead_rr_list = list(
    range(one_lead_v_low, one_lead_v_high + one_lead_v_step, one_lead_v_step)), list(
    range(one_lead_r_low, one_lead_r_high + one_lead_r_step, one_lead_r_step)), list(
    range(one_lead_rr_low, one_lead_rr_high + one_lead_rr_step, one_lead_rr_step))
for i in range(int((one_lead_v_high - one_lead_v_low + one_lead_v_step) / one_lead_v_step)): one_lead_v_to_idx_dic[
    list(range(one_lead_v_low, one_lead_v_high + one_lead_v_step, one_lead_v_step))[i]] = i
for i in range(int((one_lead_r_high - one_lead_r_low + one_lead_r_step) / one_lead_r_step)): one_lead_r_to_idx_dic[
    list(range(one_lead_r_low, one_lead_r_high + one_lead_r_step, one_lead_r_step))[i]] = i
for i in range(int((one_lead_rr_high - one_lead_rr_low + one_lead_rr_step) / one_lead_rr_step)): one_lead_rr_to_idx_dic[
    list(range(one_lead_rr_low, one_lead_rr_high + one_lead_rr_step, one_lead_rr_step))[i]] = i

# NDD distribution for Double lane change lateral decision
lc_v_low, lc_v_high, lc_v_num, lc_v_to_idx_dic = 20, 40, 21, bidict()
lc_v_list = list(np.linspace(lc_v_low, lc_v_high, num=lc_v_num))
lc_rf_low, lc_rf_high, lc_rf_num, lc_rf_to_idx_dic = 0, 115, 116, bidict()
lc_rf_list = list(np.linspace(lc_rf_low, lc_rf_high, num=lc_rf_num))
lc_rrf_low, lc_rrf_high, lc_rrf_num, lc_rrf_to_idx_dic = -10, 8, 19, bidict()
lc_rrf_list = list(np.linspace(lc_rrf_low, lc_rrf_high, num=lc_rrf_num))
lc_re_low, lc_re_high, lc_re_num, lc_re_to_idx_dic = 0, 115, 116, bidict()
lc_re_list = list(np.linspace(lc_re_low, lc_re_high, num=lc_re_num))
lc_rre_low, lc_rre_high, lc_rre_num, lc_rre_to_idx_dic = -10, 8, 19, bidict()
lc_rre_list = list(np.linspace(lc_rre_low, lc_rre_high, num=lc_rre_num))
for i in range(lc_v_num): lc_v_to_idx_dic[list(np.linspace(lc_v_low, lc_v_high, num=lc_v_num))[i]] = i
for i in range(lc_rf_num): lc_rf_to_idx_dic[list(np.linspace(lc_rf_low, lc_rf_high, num=lc_rf_num))[i]] = i
for i in range(lc_re_num): lc_re_to_idx_dic[list(np.linspace(lc_re_low, lc_re_high, num=lc_re_num))[i]] = i
for i in range(lc_rrf_num): lc_rrf_to_idx_dic[list(np.linspace(lc_rrf_low, lc_rrf_high, num=lc_rrf_num))[i]] = i
for i in range(lc_rre_num): lc_rre_to_idx_dic[list(np.linspace(lc_rre_low, lc_rre_high, num=lc_rre_num))[i]] = i

# others
ignore_adj_veh_prob, min_r_ignore = 1e-2, 5  # Probability of the ignoring the vehicle in the adjacent lane
LANE_CHANGE_SCALING_FACTOR = 1.  # ! Not used
Stochastic_IDM_threshold = 1e-10  # In the longitudinal distribution generated by the stochastic IDM model, only maneuvers with probability larger than this threshold (1e-10) will be considered as available maneuver candidates, this will filter some extremely small probabilities (e.g., 1e-20)
longi_safety_buffer, lateral_safety_buffer = 2, 2
round_rule = "Round_to_closest"
enable_One_lead_LC = True
enable_Single_LC = True
enable_Double_LC = True
enable_Cut_in_LC = True
enable_MOBIL = True
OL_LC_low_speed_flag, OL_LC_low_speed_use_v = False, 24  # ! Not used
safety_guard_enabled_flag_IDM = True
LANE_CHANGE_INDEX_LIST = [0, 1, 2]

num_acc = int(1 + ((acc_high - acc_low) / acc_resolution))
CAV_acc_low, CAV_acc_high, CAV_acc_step = -4, 2, 0.2
num_CAV_acc = int((CAV_acc_high - CAV_acc_low) / CAV_acc_step + 1)
CAV_acc_to_idx_dic = bidict()
for i in range(num_CAV_acc): CAV_acc_to_idx_dic[
    list(np.arange(CAV_acc_low, CAV_acc_high + CAV_acc_step, CAV_acc_step))[i]] = i
acc_to_idx_dic = bidict()
for m in range(num_acc): acc_to_idx_dic[list(np.linspace(acc_low, acc_high, num=num_acc))[m]] = m


def static_get_ndd_pdf(obs=None, cav_obs=None):
    _, longi_pdf = Longitudinal_NDD(obs)
    _, _, lateral_pdf = Lateral_NDD(obs)
    total_pdf = [lateral_pdf[0], lateral_pdf[2]] + \
                list(lateral_pdf[1] * longi_pdf)
    return longi_pdf, lateral_pdf, total_pdf


def Longitudinal_NDD(ego_vehicle, other_vehicles):
    """
    Decide the Longitudinal acceleration
    Input: observation of surrounding vehicles
    Output: Acceleration
    """
    if not list(CF_pdf_array):
        assert ("No CF_pdf_array file!")
    if not list(FF_pdf_array):
        assert ("No FF_pdf_array file!")
    acc = 0
    v = ego_vehicle.velocity
    f1, _ = veh.find_nearest_vehicle(ego_vehicle, other_vehicles, ego_vehicle.lane_id)

    if not f1:  # No vehicle ahead. Then FF
        round_speed, round_speed_idx = round_to_(v, round_item="speed", round_to_closest=v_resolution)
        pdf_array = FF_pdf_array[round_speed_idx]
        return acc, pdf_array

    else:  # Has vehcile ahead. Then CF
        r = f1.space - ego_vehicle.space
        rr = f1.velocity - v
        round_speed, round_speed_idx = round_to_(v, round_item="speed", round_to_closest=v_resolution)
        round_r, round_r_idx = round_to_(r, round_item="range", round_to_closest=r_resolution)
        round_rr, round_rr_idx = round_to_(rr, round_item="range_rate", round_to_closest=rr_resolution)

        if not _check_bound_constraints(r, r_low, r_high) or not _check_bound_constraints(
                rr, rr_low, rr_high) or not _check_bound_constraints(v, v_low, v_high):
            # if current state is out of the bound of the data, then use stochastic IDM to provide longitudinal deicison pdf
            pdf_array = stochastic_IDM(ego_vehicle, f1)
            if safety_guard_enabled_flag_IDM:
                pdf_array = _check_longitudinal_safety(ego_vehicle, f1, pdf_array)
            return acc, pdf_array

        pdf_array = CF_pdf_array[round_r_idx,
        round_rr_idx, round_speed_idx]
        if sum(pdf_array) == 0:
            # no CF data at this point, then use stochastic IDM to provide longitudinal deicison pdf
            pdf_array = stochastic_IDM(ego_vehicle, f1)
            if safety_guard_enabled_flag_IDM:
                pdf_array = _check_longitudinal_safety(ego_vehicle, f1, pdf_array)
            return acc, pdf_array
        return acc, pdf_array


def _check_longitudinal_safety(ego_vehicle, front_vehicle, pdf_array, lateral_result=None, CAV_flag=False):
    """Check longitudinal safety for vehicle.

    Args:
        obs (dict): Processed observation of the vehicle.
        pdf_array (list(float)): Old possibility distribution of the maneuvers.
        lateral_result (list(float), optional): Possibility distribution of the lateral maneuvers. Defaults to None.
        CAV_flag (bool, optional): Check whether the vehicle is the CAV. Defaults to False.

    Returns:
        list(float): New possibility distribution of the maneuvers after checking the longitudinal direction.
    """
    safety_buffer = longi_safety_buffer
    for i in range(len(pdf_array) - 1, -1, -1):
        if CAV_flag:
            acc = CAV_acc_to_idx_dic.inverse[i]
        else:
            acc = acc_to_idx_dic.inverse[i]
        if front_vehicle is not None:
            rr = front_vehicle.velocity - ego_vehicle.velocity
            r = front_vehicle.space - ego_vehicle.space
            criterion_1 = rr + r + 0.5 * (acc_low - acc)
            self_v_2, f_v_2 = max(ego_vehicle.velocity + acc, v_low), max((front_vehicle.velocity + acc_low), v_low)
            dist_r = (self_v_2 ** 2 - v_low ** 2) / (2 * abs(acc_low))
            dist_f = (f_v_2 ** 2 - v_low ** 2) / (2 * abs(acc_low)) + v_low * (f_v_2 - self_v_2) / acc_low
            criterion_2 = criterion_1 - dist_r + dist_f
            if criterion_1 <= safety_buffer or criterion_2 <= safety_buffer:
                pdf_array[i] = 0
            else:
                break

    # Only set the decelerate most when none of lateral is OK.
    if lateral_result is not None:
        lateral_feasible = lateral_result[0] or lateral_result[2]
    else:
        lateral_feasible = False
    if np.sum(pdf_array) == 0 and not lateral_feasible:
        pdf_array[0] = 1 if not CAV_flag else np.exp(-2)
        return pdf_array

    if CAV_flag:
        new_pdf_array = pdf_array
    else:
        new_pdf_array = pdf_array / np.sum(pdf_array)
    return new_pdf_array


def stochastic_IDM(ego_vehicle, front_vehicle):
    tmp_acc = veh.Vehicle.IDM_acceleration(ego_vehicle.velocity, front_vehicle.velocity,
                                       front_vehicle.space - ego_vehicle.space, "BV")
    tmp_acc = np.clip(tmp_acc, acc_low, acc_high)
    acc_possi_list = scipy.stats.norm.pdf(acc_list, tmp_acc, 0.3)
    # clip the possibility to avoid too small possibility
    acc_possi_list = [val if val > Stochastic_IDM_threshold else 0 for val in acc_possi_list]
    assert (sum(acc_possi_list) > 0)
    acc_possi_list = acc_possi_list / (sum(acc_possi_list))
    return acc_possi_list


def Lateral_NDD(ego_vehicle, other_vehicles):
    """
    Decide the Lateral movement
    Input: observation of surrounding vehicles
    Output: whether do lane change (True, False), lane_change_idx (0:Left, 1:Still, 2:Right), action_pdf
    """
    initial_pdf = np.array([0, 1, 0])  # Left, Still, Right
    if not list(OL_pdf):
        raise ValueError("No One Lead pdf file!")

    lane_id, v = ego_vehicle.lane_id, ego_vehicle.velocity

    f1, r1, f0, r0, f2, r2 = None, None, None, None, None, None
    if lane_id > 0:
        f0, r0 = veh.find_nearest_vehicle(ego_vehicle, other_vehicles, lane_id - 1)
    f1, r1 = veh.find_nearest_vehicle(ego_vehicle, other_vehicles, lane_id)
    if lane_id < 2:
        f2, r2 = veh.find_nearest_vehicle(ego_vehicle, other_vehicles, lane_id + 1)

    if not f1:  # No vehicle ahead
        return False, 1, initial_pdf
    else:  # Has vehcile ahead
        left_prob, still_prob, right_prob = 0, 0, 0
        LC_related_list = []
        LC_type_list = []

        # Check NDD LC probability on both sides. Used to determine whether use MOBIL
        for item in ["Left", "Right"]:
            if item == "Left":
                surrounding = (f1, f0, r0)
                left_prob, LC_type, LC_related = _LC_prob(surrounding, ego_vehicle)
                LC_related_list.append(LC_related)
                LC_type_list.append(LC_type)
            else:
                surrounding = (f1, f2, r2)
                right_prob, LC_type, LC_related = _LC_prob(surrounding, ego_vehicle)
                LC_related_list.append(LC_related)
                LC_type_list.append(LC_type)
        has_LC_data_on_at_least_one_side_flag = True
        if left_prob is None and right_prob is None:
            has_LC_data_on_at_least_one_side_flag = False
        # If there is data on at least one side, then other side LC prob=0 if there is no LC data on this side
        if has_LC_data_on_at_least_one_side_flag:
            if left_prob is None:
                left_prob = 0
                right_prob = 2 * right_prob
            elif right_prob is None:
                right_prob = 0
                left_prob = 2 * left_prob
        # Check whether there is CF data in this situation, if not or has no LC data on both sides then use stochastic MOBIL
        has_CF_data_flag = check_whether_has_CF_data(ego_vehicle, f1)
        MOBIL_flag = ((not has_CF_data_flag) and (np.floor(v + 0.5) <= 21)) or (
            not has_LC_data_on_at_least_one_side_flag)

        # MOBIL
        if MOBIL_flag:
            left_prob, right_prob = veh.Vehicle.MOBIL_lane_changing(ego_vehicle, other_vehicles, "BV")

        # In the leftest or rightest, double the other lane change probability
        if lane_id == 0:
            left_prob = 0
        if lane_id == 2:
            right_prob = 0
        if left_prob + right_prob > 1:
            tmp = left_prob + right_prob
            left_prob *= 0.9 / tmp
            right_prob *= 0.9 / tmp
        still_prob = 1 - left_prob - right_prob
        pdf_array = np.array([left_prob, still_prob, right_prob])

        lane_change_idx = np.random.choice(LANE_CHANGE_INDEX_LIST, None, False, pdf_array)
        to_lane_id = lane_id + lane_change_idx - 1
        if lane_change_idx != 1:
            return True, lane_change_idx, pdf_array
        else:
            return False, lane_change_idx, pdf_array


def check_whether_has_CF_data(ego_vehicle, front_vehicle):
    """
    If there is no CF data, then use IDM+MOBIL
    """
    v = ego_vehicle.velocity
    r = front_vehicle.space - ego_vehicle.space
    rr = front_vehicle.velocity - v
    round_speed, round_speed_idx = round_to_(v, round_item="speed", round_to_closest=v_resolution)
    round_r, round_r_idx = round_to_(r, round_item="range", round_to_closest=r_resolution)
    round_rr, round_rr_idx = round_to_(rr, round_item="range_rate", round_to_closest=rr_resolution)

    pdf_array = CF_pdf_array[round_r_idx,
    round_rr_idx, round_speed_idx]
    if sum(pdf_array) == 0:
        return False
    else:
        return True


def _LC_prob(surrounding_vehicles, ego_vehicle):
    """
    Input: (veh_front, veh_adj_front, veh_adj_back)
    output: the lane change probability and the expected lane change probability (take the ignored situation into account)
    """
    LC_prob, E_LC_prob = None, None
    veh_front, veh_adj_front, veh_adj_rear = surrounding_vehicles

    if not veh_adj_front and not veh_adj_rear:
        # One lead LC
        LC_prob, LC_related = _get_One_lead_LC_prob(veh_front, ego_vehicle)
        E_LC_prob = LC_prob
        return E_LC_prob, "One_lead", LC_related

    elif veh_adj_front and not veh_adj_rear:
        # Single lane change
        LC_prob, LC_related = _get_Single_LC_prob(veh_front, veh_adj_front, ego_vehicle)
        E_LC_prob = LC_prob
        return E_LC_prob, "SLC", LC_related

    elif not veh_adj_front and veh_adj_rear:
        # One Lead prob
        OL_LC_prob, OL_LC_related = _get_One_lead_LC_prob(veh_front, ego_vehicle)

        # Cut in prob
        CI_LC_prob, CI_LC_related = _get_Cut_in_LC_prob(veh_front, veh_adj_rear, ego_vehicle)
        LC_related = CI_LC_related

        r_adj = ego_vehicle.space - veh_adj_rear.space

        if (r_adj >= min_r_ignore) and (CI_LC_prob is not None) and (OL_LC_prob is not None):
            E_LC_prob = ignore_adj_veh_prob * OL_LC_prob + (1 - ignore_adj_veh_prob) * CI_LC_prob
        else:
            E_LC_prob = CI_LC_prob
        return E_LC_prob, "Cut_in", LC_related

    elif veh_adj_front and veh_adj_rear:
        # Single lane change prob
        SLC_LC_prob, SLC_LC_related = _get_Single_LC_prob(veh_front, veh_adj_front, ego_vehicle)

        # Double lane change prob
        DLC_LC_prob, DLC_LC_related = _get_Double_LC_prob(veh_adj_front, veh_adj_rear, ego_vehicle)
        LC_related = DLC_LC_related

        r_adj = ego_vehicle.space - veh_adj_rear.space

        if (r_adj >= min_r_ignore) and (DLC_LC_prob is not None) and (SLC_LC_prob is not None):
            E_LC_prob = ignore_adj_veh_prob * SLC_LC_prob + (1 - ignore_adj_veh_prob) * DLC_LC_prob
        else:
            E_LC_prob = DLC_LC_prob
        return E_LC_prob, "DLC", LC_related


def round_value_lane_change(real_value, value_list, round_item="speed"):
    if real_value < value_list[0]:
        real_value = value_list[0]
    elif real_value > value_list[-1]:
        real_value = value_list[-1]

    if round_rule == "Round_to_closest":
        min_val, max_val, resolution = value_list[0], value_list[-1], value_list[1] - value_list[0]
        # real_value_old = np.clip(round((real_value - min_val) / resolution)*resolution + min_val, min_val, max_val)
        _num = (real_value - min_val) / resolution
        if int(_num * 2) == _num * 2:
            if int(_num) % 2 != 0:
                _num += 0.5
        else:
            _num += 0.5
        real_value_new = int(_num) * resolution + min_val
        # assert real_value_new==real_value_old
        real_value = real_value_new

    if round_item == "speed":
        value_idx = bisect.bisect_left(value_list, real_value)
        value_idx = value_idx if real_value <= value_list[-1] else value_idx - 1
        try:
            assert value_idx <= (len(value_list) - 1)
            assert value_idx >= 0
        except:
            print("Error in lane change round value")
        round_value = value_list[value_idx]
        return round_value, value_idx
    else:
        value_idx = bisect.bisect_left(value_list, real_value)
        value_idx = value_idx - \
                    1 if real_value != value_list[value_idx] else value_idx
        try:
            assert value_idx <= (len(value_list) - 1)
            assert value_idx >= 0
        except:
            print("Error in lane change round value")
        round_value = value_list[value_idx]
        return round_value, value_idx


def _check_bound_constraints(value, bound_low, bound_high):
    if value < bound_low or value > bound_high:
        return False
    else:
        return True


def round_value_lane_change(real_value, value_list, round_item="speed"):
    if real_value < value_list[0]:
        real_value = value_list[0]
    elif real_value > value_list[-1]:
        real_value = value_list[-1]

    if round_rule == "Round_to_closest":
        min_val, max_val, resolution = value_list[0], value_list[-1], value_list[1] - value_list[0]
        # real_value_old = np.clip(round((real_value - min_val) / resolution)*resolution + min_val, min_val, max_val)
        _num = (real_value - min_val) / resolution
        if int(_num * 2) == _num * 2:
            if int(_num) % 2 != 0:
                _num += 0.5
        else:
            _num += 0.5
        real_value_new = int(_num) * resolution + min_val
        # assert real_value_new==real_value_old
        real_value = real_value_new

    if round_item == "speed":
        value_idx = bisect.bisect_left(value_list, real_value)
        value_idx = value_idx if real_value <= value_list[-1] else value_idx - 1
        try:
            assert value_idx <= (len(value_list) - 1)
            assert value_idx >= 0
        except:
            print("Error in lane change round value")
        round_value = value_list[value_idx]
        return round_value, value_idx
    else:
        value_idx = bisect.bisect_left(value_list, real_value)
        value_idx = value_idx - 1 if real_value != value_list[value_idx] else value_idx
        try:
            assert value_idx <= (len(value_list) - 1)
            assert value_idx >= 0
        except:
            print("Error in lane change round value")
        round_value = value_list[value_idx]
        return round_value, value_idx


# @profile
def _get_One_lead_LC_prob(veh_front, ego_vehicle):
    v = ego_vehicle.velocity
    if not enable_One_lead_LC:
        return 0, None
    r, rr = veh_front.space - ego_vehicle.space, veh_front.velocity - v
    # Check bound
    if (not _check_bound_constraints(v, one_lead_v_low, one_lead_v_high)
            or not _check_bound_constraints(r, one_lead_r_low, one_lead_r_high)
            or not _check_bound_constraints(rr, one_lead_rr_low, one_lead_rr_high)):
        return 0, None

    round_r, round_r_idx = round_value_lane_change(real_value=r, value_list=one_lead_r_list)
    round_rr, round_rr_idx = round_value_lane_change(real_value=rr, value_list=one_lead_rr_list)
    round_speed, round_speed_idx = round_value_lane_change(real_value=v, value_list=one_lead_speed_list,
                                                           round_item="speed")
    # Since currently the OL raw data v>=24. So for v<=23, there is definitely no LC, so use the v==24 data when v<=23
    if round_speed <= 23 and OL_LC_low_speed_flag:
        v_diff = OL_LC_low_speed_use_v - round_speed
        assert (v_diff > 0)
        round_rr = round_rr - v_diff
        round_rr, round_rr_idx = round_value_lane_change(real_value=round_rr, value_list=one_lead_rr_list)
        round_speed, round_speed_idx = round_value_lane_change(real_value=OL_LC_low_speed_use_v,
                                                               value_list=one_lead_speed_list, round_item="speed")

    lane_change_prob = OL_pdf[round_speed_idx, round_r_idx, round_rr_idx, :][0] * LANE_CHANGE_SCALING_FACTOR
    LC_related = (v, r, rr, round_speed, round_r, round_rr)

    # chech whether there is LC data in this case
    if OL_pdf[round_speed_idx, round_r_idx, round_rr_idx, :][0] == 0 and \
            OL_pdf[round_speed_idx, round_r_idx, round_rr_idx, :][1] == 0:
        lane_change_prob = None

    return lane_change_prob, LC_related


# @profile
def _get_Double_LC_prob(veh_adj_front, veh_adj_rear, ego_vehicle):
    v = ego_vehicle.velocity
    v_list, r1_list, r2_list, rr1_list, rr2_list = lc_v_list, lc_rf_list, lc_re_list, lc_rrf_list, lc_rre_list
    LC_related = None
    # Double lane change
    if not enable_Double_LC:
        return 0, LC_related
    r1, rr1 = veh_adj_front.space - ego_vehicle.space, veh_adj_front.velocity - v
    r2, rr2 = ego_vehicle.space - veh_adj_rear.space, v - veh_adj_rear.velocity
    if not _check_bound_constraints(v, lc_v_low, lc_v_high):
        return 0, LC_related
    elif not _check_bound_constraints(r1, lc_rf_low, lc_rf_high):
        return 0, LC_related
    elif not _check_bound_constraints(rr1, lc_rrf_low, lc_rrf_high):
        return 0, LC_related
    elif not _check_bound_constraints(r2, lc_re_low, lc_re_high):
        return 0, LC_related
    elif not _check_bound_constraints(rr2, lc_rre_low, lc_rre_high):
        return 0, LC_related
    round_v, v_idx = round_value_lane_change(
        real_value=v, value_list=v_list, round_item="speed")
    round_r1, r1_idx = round_value_lane_change(
        real_value=r1, value_list=r1_list)
    round_rr1, rr1_idx = round_value_lane_change(
        real_value=rr1, value_list=rr1_list)
    round_r2, r2_idx = round_value_lane_change(
        real_value=r2, value_list=r2_list)
    round_rr2, rr2_idx = round_value_lane_change(
        real_value=rr2, value_list=rr2_list)

    lane_change_prob = DLC_pdf[v_idx, r1_idx, rr1_idx, r2_idx, rr2_idx, :][0] * LANE_CHANGE_SCALING_FACTOR

    LC_related = (v, r1, rr1, r2, rr2, round_v, round_r1, round_rr1, round_r2, round_rr2, lane_change_prob)

    # chech whether there is LC data in this case
    if DLC_pdf[v_idx, r1_idx, rr1_idx, r2_idx, rr2_idx, :][0] == 0 and \
            DLC_pdf[v_idx, r1_idx, rr1_idx, r2_idx, rr2_idx, :][1] == 0:
        lane_change_prob = None

    return lane_change_prob, LC_related


# @profile
def _get_Single_LC_prob(veh_front, veh_adj_front, ego_vehicle):
    v = ego_vehicle.velocity
    v_list, r1_list, r2_list, rr1_list, rr2_list = lc_v_list, lc_rf_list, lc_re_list, lc_rrf_list, lc_rre_list
    LC_related = None
    # Single lane change
    if not enable_Single_LC:
        return 0, LC_related

    r1, rr1 = veh_front.space - ego_vehicle.space, veh_front.velocity - v
    r2, rr2 = veh_adj_front.space - ego_vehicle.space, veh_adj_front.velocity - v

    if not _check_bound_constraints(v, lc_v_low, lc_v_high):
        return 0, LC_related
    elif not _check_bound_constraints(r1, lc_rf_low, lc_rf_high):
        return 0, LC_related
    elif not _check_bound_constraints(rr1, lc_rrf_low, lc_rrf_high):
        return 0, LC_related
    elif not _check_bound_constraints(r2, lc_rf_low, lc_rf_high):
        return 0, LC_related
    elif not _check_bound_constraints(rr2, lc_rrf_low, lc_rrf_high):
        return 0, LC_related

    round_v, v_idx = round_value_lane_change(
        real_value=v, value_list=v_list, round_item="speed")
    round_r1, r1_idx = round_value_lane_change(
        real_value=r1, value_list=r1_list)
    round_rr1, rr1_idx = round_value_lane_change(
        real_value=rr1, value_list=rr1_list)
    round_r2, r2_idx = round_value_lane_change(
        real_value=r2, value_list=r2_list)
    round_rr2, rr2_idx = round_value_lane_change(
        real_value=rr2, value_list=rr2_list)

    lane_change_prob = SLC_pdf[v_idx, r1_idx, rr1_idx,
                       r2_idx, rr2_idx, :][0] * LANE_CHANGE_SCALING_FACTOR

    LC_related = (v, r1, rr1, r2, rr2, round_v, round_r1,
                  round_rr1, round_r2, round_rr2, lane_change_prob)

    # chech whether there is LC data in this case
    if SLC_pdf[v_idx, r1_idx, rr1_idx, r2_idx, rr2_idx, :][0] == 0 and \
            SLC_pdf[v_idx, r1_idx, rr1_idx, r2_idx, rr2_idx, :][1] == 0:
        lane_change_prob = None

    return lane_change_prob, LC_related


# @profile
def _get_Cut_in_LC_prob(veh_front, veh_adj_rear, ego_vehicle):
    v = ego_vehicle.velocity
    v_list, r1_list, r2_list, rr1_list, rr2_list = lc_v_list, lc_rf_list, lc_re_list, lc_rrf_list, lc_rre_list
    LC_related = None

    if not enable_Cut_in_LC:
        return 0, None

    r1, rr1 = veh_front.space - ego_vehicle.space, veh_front.velocity - v
    r2, rr2 = ego_vehicle.space - veh_adj_rear.space, v - veh_adj_rear.velocity

    if not _check_bound_constraints(v, lc_v_low, lc_v_high):
        return 0, LC_related
    elif not _check_bound_constraints(r1, lc_rf_low, lc_rf_high):
        return 0, LC_related
    elif not _check_bound_constraints(rr1, lc_rrf_low, lc_rrf_high):
        return 0, LC_related
    elif not _check_bound_constraints(r2, lc_rf_low, lc_rf_high):
        return 0, LC_related
    elif not _check_bound_constraints(rr2, lc_rrf_low, lc_rrf_high):
        return 0, LC_related

    round_v, v_idx = round_value_lane_change(
        real_value=v, value_list=v_list, round_item="speed")
    round_r1, r1_idx = round_value_lane_change(
        real_value=r1, value_list=r1_list)
    round_rr1, rr1_idx = round_value_lane_change(
        real_value=rr1, value_list=rr1_list)
    round_r2, r2_idx = round_value_lane_change(
        real_value=r2, value_list=r2_list)
    round_rr2, rr2_idx = round_value_lane_change(
        real_value=rr2, value_list=rr2_list)

    lane_change_prob = CI_pdf[v_idx, r1_idx, rr1_idx,
                       r2_idx, rr2_idx, :][0] * LANE_CHANGE_SCALING_FACTOR

    LC_related = (v, r1, rr1, r2, rr2, round_v, round_r1,
                  round_rr1, round_r2, round_rr2, lane_change_prob)

    # chech whether there is LC data in this case
    if CI_pdf[v_idx, r1_idx, rr1_idx, r2_idx, rr2_idx, :][0] == 0 and \
            CI_pdf[v_idx, r1_idx, rr1_idx, r2_idx, rr2_idx, :][1] == 0:
        lane_change_prob = None

    return lane_change_prob, LC_related


# @profile
def round_to_(val, round_item, round_to_closest):
    """
    round the val to the round_to_closest (for example 1.0, 0.2 ...)
    """
    if round_item == "speed":
        value_list = speed_list
    elif round_item == "range":
        value_list = r_list
    elif round_item == "range_rate":
        value_list = rr_list

    if round_to_closest == 1:
        mul, add, check = 1, 1, 0.5
    elif round_to_closest == 0.5:
        mul, add, check = 2, 0.5, 0.25
    elif round_to_closest == 0.2:
        mul, add, check = 5, 0.2, 0.1

    if val < value_list[0]:
        val = value_list[0]
    elif val > value_list[-1]:
        val = value_list[-1]

    if round_rule == "Round_to_closest":
        round_val = np.floor(val * mul + 0.5) / mul
        try:
            assert (-check < round_val - val <= check + 1e-10)
        except:
            round_val += add
            try:
                assert (-check < round_val - val <= check + 1e-10)
            except:
                print(val, round_val)
                raise ValueError("Round error!")

    try:
        round_idx = value_list.index(round_val)
    except:
        round_idx = min(range(len(value_list)),
                        key=lambda i: abs(value_list[i] - round_val))
        assert (np.abs(value_list[round_idx] - round_val) < 1e-8)

    return round_val, round_idx
