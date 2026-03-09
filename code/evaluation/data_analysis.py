import matplotlib.pyplot as plt
import os
import numpy as np


def plot_distribution(s_values, v_values, v_lead_values):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(s_values, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('s values')
    plt.ylabel('Frequency')
    plt.title('Distribution of s values for TTC (1.25, 1.5)')

    plt.subplot(1, 3, 2)
    plt.hist(v_values, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('v values')
    plt.ylabel('Frequency')
    plt.title('Distribution of v values for TTC (1.25, 1.5)')

    plt.subplot(1, 3, 3)
    plt.hist(v_lead_values, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('v_lead values')
    plt.ylabel('Frequency')
    plt.title('Distribution of v_lead values for TTC (1.25, 1.5)')

    plt.tight_layout()
    plt.show()


from scipy.stats import gaussian_kde


def plot_with_kde(ax, data, weights, color, xlabel, ylabel, title, ttc_category):
    # 绘制频数直方图
    counts, bins, _ = ax.hist(data, bins=50, color=color, edgecolor='black', alpha=0.7, weights=weights)

    # 计算概率密度并绘制曲线
    kde = gaussian_kde(data, weights=weights)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    density = kde(bin_centers)

    ax2 = ax.twinx()
    ax2.plot(bin_centers, density, color='black')
    ax2.set_ylabel('Probability Density')
    ax2.set_ylim(0.0, 0.4)
    # 设置 xlim 范围

    if xlabel == 'Distance (m)':
        upper = float(ttc_category.split('_')[2])
        if upper > 8:
            ax2.set_xlim(0, 100)
        elif upper >= 6.0:
            ax2.set_xlim(0, 60)
        elif upper >= 2:
            ax2.set_xlim(0, 30)
        elif upper >= 1:
            ax2.set_xlim(0, 15)
        else:
            ax2.set_xlim(0, 8)
    else:
        upper = float(ttc_category.split('_')[2])
        if upper >= 8:
            ax2.set_xlim(0, 50)
        else:
            ax2.set_xlim(0, 30)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_title(title)


def plot_state_distribution(states, title, issave=False, output_dir=None):
    """Plot the distribution of states (v_lead, v, s)."""
    v_lead = [state[0] for state in states.keys()]
    # if state[1] > 29 else state[1]*state[1]/60if state[0] > 29 else state[0]*state[0]/30*state[2]/80
    v = [state[1] for state in states.keys()]
    s = [state[2] for state in states.keys()]
    weights = list(states.values())
    # weights = [0.5] * len(states)
    plt.figure(figsize=(15, 5))

    ax1 = plt.subplot(1, 3, 1)
    plot_with_kde(ax1, s, weights, '#82B0D2', 'Distance (m)', 'Weighted Frequency', f'Distribution of s', title)

    ax2 = plt.subplot(1, 3, 2)
    plot_with_kde(ax2, v, weights, '#FFBE7A', 'Ego vehicle speed (m/s)', 'Weighted Frequency', f'Distribution of v',
                  title)

    ax3 = plt.subplot(1, 3, 3)
    plot_with_kde(ax3, v_lead, weights, '#FA7F6F', 'Lead vehicle speed (m/s)', 'Weighted Frequency',
                  f'Distribution of v_lead', title)

    # plt.suptitle(title)
    plt.tight_layout()
    # plt.show()
    if issave:
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, title.replace('.', '_') + '.png'))
        plt.close()


def plot_collision_severity_distribution(severities, weights, title, output_dir=None, issave=False):
    """Plot the weighted distribution of collision severities."""
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Define the bin edges with a fixed width
    bin_width = 2000
    bins = np.arange(min(severities), max(severities) + bin_width, bin_width)

    # Now pass the bins array to ax.hist
    counts, bins, _ = ax.hist(severities, bins=bins, color='#82B0D2', edgecolor='black', alpha=0.7, weights=weights)
    # Calculate and plot KDE
    kde = gaussian_kde(severities, weights=weights)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    density = kde(bin_centers)

    ax2 = ax.twinx()
    ax2.plot(bin_centers, density, color='black')
    ax2.set_ylabel('Probability Density')
    ax2.set_xlim(0, 40000)
    ax.set_xlabel('Collision Severity (Joules)')
    ax.set_ylabel('Weighted Frequency')
    ax.set_title(title)

    plt.tight_layout()

    if issave:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, title.replace('.', '_') + '.png'))
        plt.close()
    else:
        plt.show()
        a = 1
