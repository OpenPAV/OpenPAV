import numpy as np
import itertools
import random
from vehicles import Vehicle, generate_initial_state
from utils import *
from datetime import datetime
import ast

state_weights = {}


def discretize(value):
    epsilon = 1e-5
    return round((value + epsilon) / granularity) * granularity


def _reborn_state(mode):
    """Return the configured reborn state for a given mode."""
    return generate_initial_state(mode)


def _crash_state(mode):
    """Return a dedicated crash state representation."""
    if mode in ("CF", "IDM"):
        # Keep TTC strictly negative for crash bucketing.
        return Vehicle(1, 0, 0), Vehicle(0, -1, 0)
    return _reborn_state(mode)


def _is_out_of_domain(v_next, v_lead_next, s_next):
    """Check if the candidate state is outside configured simulation domain."""
    return (
        v_next < min_speed
        or v_next > max_speed
        or v_lead_next < min_speed
        or v_lead_next > max_speed
        or s_next > max_space
    )


def sample_states_uniform(ttc_range, num_samples):
    """Sample initial states uniformly from the provided distribution, avoiding extra memory usage."""
    global state_weights
    state_weights = {state: prob for state, prob in state_weights.items() if state[-1] < ttc_range[1]}
    filtered_states_weight = {
        state: prob for state, prob in state_weights.items() if ttc_range[0] <= state[-1] < ttc_range[1]
    }
    if not filtered_states_weight:
        return {}
    population = list(filtered_states_weight.items())
    sample_size = min(num_samples, len(population))
    sampled_states_with_prob = dict(random.sample(population, sample_size))
    return sampled_states_with_prob


def sample_states_sequential(ttc_range, num_samples, start_idx=0):
    """Sample a contiguous slice of states in the TTC bucket, without randomness."""
    global state_weights
    state_weights = {state: prob for state, prob in state_weights.items() if state[-1] < ttc_range[1]}
    filtered_states_weight = {
        state: prob for state, prob in state_weights.items() if ttc_range[0] <= state[-1] < ttc_range[1]
    }
    if not filtered_states_weight:
        return {}, start_idx, True

    population = list(filtered_states_weight.items())
    if start_idx >= len(population):
        return {}, start_idx, True

    end_idx = min(start_idx + num_samples, len(population))
    sampled_states_with_prob = dict(population[start_idx:end_idx])
    exhausted = end_idx >= len(population)
    return sampled_states_with_prob, end_idx, exhausted


def analyze_sampled_states(sampled_states, ttc_range, mode="CF"):
    """Analyze sampled states and their transitions.

    Returns:
        transition_counts: weighted transition accumulation.
        crash_hits: number of sampled source states that have non-zero crash transition.
    """

    from_category = f"{ttc_range[0]}-{ttc_range[1]}"
    transition_counts = {}
    transition_counts[from_category] = {}
    crash_hits = 0
    for state, father_prob in sampled_states.items():
        next_states_probs = next_state_probabilities(state[0], mode)  # Use state without TTC
        state_has_crash = False
        for next_state, prob in next_states_probs:
            if next_state is None:
                continue
            ttc_next = calculate_ttc(next_state, mode)
            ttc_category_next = get_ttc_category(ttc_next)
            next_state_with_ttc = (next_state, ttc_next)  # Add TTC to the state tuple

            # Record the transition from current TTC to next TTC category
            if ttc_category_next in transition_counts[from_category]:
                transition_counts[from_category][ttc_category_next] += prob * father_prob
            else:
                transition_counts[from_category][ttc_category_next] = prob * father_prob
            if ttc_next < ttc_range[0]:
                if next_state_with_ttc in state_weights:
                    state_weights[next_state_with_ttc] += prob * father_prob
                else:
                    state_weights[next_state_with_ttc] = prob * father_prob
            if ttc_next < 0 and prob > 0:
                state_has_crash = True
        if state_has_crash:
            crash_hits += 1

    return transition_counts, crash_hits


def next_state_probabilities(current_state, mode="CF", dt=DT, threshold=1e-20):
    """Calculate the next state probabilities for a given current state."""
    next_states_dict = {}

    if mode in ("CF", "IDM"):
        FAV = current_state[0]
        LBV = current_state[1]
        initial_state = _reborn_state(mode)
        crash_state = _crash_state(mode)

        # If current state is already a crash, reborn immediately.
        if LBV.space <= 0:
            next_states_dict[initial_state] = 1.0
            return list(next_states_dict.items())

        # Define a range for next possible accelerations and lead vehicle speeds
        acc_range_AV, acc_distribution_AV = FAV.get_action_distributions_AV(LBV, mode)
        acc_changes_BV, acc_distribution_BV = LBV.get_action_distributions_BV(FAV, [], mode)

        # Handle both continuous (scipy.stats) and discrete distributions for AV
        if hasattr(acc_distribution_AV, "pdf"):
            acc_probs_AV = np.array([acc_distribution_AV.pdf(a) * action_granularity for a in acc_range_AV])
        else:
            acc_probs_AV = np.array(acc_distribution_AV, dtype=float)

        for acc_FAV, prob_acc in zip(acc_range_AV, acc_probs_AV):
            for acc_LBV in acc_changes_BV:
                v_lead_next = LBV.velocity + acc_LBV * dt
                v_next = FAV.velocity + acc_FAV * dt
                s_next = LBV.space + (LBV.velocity - FAV.velocity) * dt

                # Calculate the probability of this transition
                prob_v_lead = acc_distribution_BV.pdf(acc_LBV) * action_granularity
                prob = prob_acc * prob_v_lead

                if prob < threshold:
                    continue

                # Crash handling: transition to dedicated crash state.
                if s_next <= 0:
                    next_states_dict[crash_state] = next_states_dict.get(crash_state, 0.0) + prob
                    continue

                if _is_out_of_domain(v_next, v_lead_next, s_next):
                    v_lead_next = initial_state[1].velocity
                    v_next = initial_state[0].velocity
                    s_next = initial_state[1].space

                # Discretize the next state
                v_lead_next = discretize(v_lead_next)
                v_next = discretize(v_next)
                s_next = discretize(s_next)
                next_state = (Vehicle(v_next, 0, 0), Vehicle(v_lead_next, s_next, 0))
                if next_state in next_states_dict:
                    next_states_dict[next_state] += prob
                else:
                    next_states_dict[next_state] = prob

    # Normalize probabilities
    total_prob = sum(next_states_dict.values())
    if total_prob <= 0:
        # Fallback to reborn if probability mass was fully filtered.
        next_states_dict[initial_state] = 1.0
        total_prob = 1.0
    for state in next_states_dict:
        next_states_dict[state] /= total_prob

    return list(next_states_dict.items())


def random_sample_product(choices_list, sample_size):
    sampled = []
    count = 0
    for item in itertools.product(*choices_list):  # 逐步生成组合
        count += 1
        if len(sampled) < sample_size:
            sampled.append(item)  # 直接添加前 100 个
        else:
            return sampled
    return sampled


def sample_one_state(current_state, mode="CF", dt=DT):
    """Sample a next state based on transition probabilities."""

    if mode in ("CF", "IDM"):
        FAV = current_state[0]
        LBV = current_state[1]
        initial_state = _reborn_state(mode)
        crash_state = _crash_state(mode)

        if LBV.space <= 0:
            return initial_state

        acc_range, acc_distribution = FAV.get_action_distributions_AV(LBV, mode)
        acc_lead_changes, acc_lead_distribution = LBV.get_action_distributions_BV(FAV, [], mode)
        if len(acc_range) == 0 or len(acc_lead_changes) == 0:
            return initial_state

        # AV: handle both continuous (scipy.stats) and discrete distributions
        if hasattr(acc_distribution, "pdf"):
            p_av = np.array([acc_distribution.pdf(a) * action_granularity for a in acc_range])
        else:
            p_av = np.array(acc_distribution, dtype=float)
        if p_av.size == 0:
            return initial_state
        if p_av.sum() > 0:
            p_av = p_av / p_av.sum()
        else:
            p_av = np.ones_like(p_av) / len(p_av)
        acc_FAV = np.random.choice(acc_range, p=p_av)

        # BV: still Gaussian
        p = np.array([acc_lead_distribution.pdf(a) * action_granularity for a in acc_lead_changes])
        if p.size == 0:
            return initial_state
        p_sum = p.sum()
        if p_sum <= 0:
            p = np.ones_like(p) / len(p)
        else:
            p /= p_sum
        acc_LBV = np.random.choice(acc_lead_changes, p=p)

        v_lead_next = discretize(LBV.velocity + acc_LBV * dt)
        v_next = discretize(FAV.velocity + acc_FAV * dt)
        s_next = discretize(LBV.space + (LBV.velocity - FAV.velocity) * dt)

        if s_next <= 0:
            return crash_state

        if _is_out_of_domain(v_next, v_lead_next, s_next):
            return initial_state

        return Vehicle(v_next, 0, FAV.lane_id), Vehicle(v_lead_next, s_next, LBV.lane_id)


def sample_states_weighted(states_weight, num_samples):
    """Sample initial states with TTC > 8 based on the provided distribution."""
    states_in_range = list(states_weight.keys())
    weights = list(states_weight.values())
    probabilities = np.array(weights) / sum(weights)
    sampled_indices = np.random.choice(len(states_in_range), num_samples, p=probabilities)
    sampled_states_with_prob = {states_in_range[i]: probabilities[i] for i in sampled_indices}
    return sampled_states_with_prob


def initialize_from_file(initial_state_weights_file, ttc_range, mode="CF"):
    """Initialize state weights from the file."""
    initial_states = load_from_json(initial_state_weights_file)
    if mode in ("CF", "IDM"):
        initial_states = {int(float(key)): value for key, value in initial_states.items()}
    total_state_number = sum(initial_states.values())

    initial_state_weights = {}
    for key, count in initial_states.items():
        state = unhash_state(key, mode)
        ttc = calculate_ttc(state, mode)
        state_with_ttc = (state, ttc)  # tuple
        if ttc_range[0] <= ttc < ttc_range[1]:
            initial_state_weights[state_with_ttc] = count / total_state_number
    del initial_states
    return initial_state_weights
