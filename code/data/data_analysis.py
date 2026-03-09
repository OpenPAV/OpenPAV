import numpy as np
from PIL import Image
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def analyze_statistics(input_path, output_path, has_kde=True):
    """Analyze statistical data and generate histograms for selected columns."""
    df = pd.read_csv(input_path)

    # Define the columns for statistical analysis
    columns_to_describe = ['Speed_LV', 'Acc_LV', 'Speed_FAV', 'Acc_FAV', 'Spatial_Gap', 'Spatial_Headway']
    statistics = df[columns_to_describe].describe()
    statistics.to_csv(output_path + '_statistics.csv')

    # Define columns to generate histograms
    columns_to_check = ['Speed_FAV', 'Acc_FAV', 'Spatial_Gap', 'Speed_Diff']
    for column in columns_to_check:
        plt.figure(figsize=(18, 14))
        plt.tick_params(axis='x', labelsize=40)  # Set font size for x-axis labels
        plt.tick_params(axis='y', labelsize=40)  # Set font size for y-axis labels
        if column == 'Acc_FAV':
            plt.title(f'Distribution of $a$', fontsize=50)
            sns.histplot(df[column], kde=has_kde, color=(42 / 255, 157 / 255, 140 / 255), line_kws={'linewidth': 5})
            plt.xlabel('$a$ ($m/s^2$)', fontsize=50)
        elif column == 'Spatial_Gap':
            plt.title(f'Distribution of $s$', fontsize=50)
            sns.histplot(df[column], kde=has_kde, color=(233 / 255, 196 / 255, 107 / 255), line_kws={'linewidth': 5})
            plt.xlabel('$s$ ($m$)', fontsize=50)
        elif column == 'Speed_Diff':
            plt.title(f'Distribution of $\Delta v$', fontsize=50)
            sns.histplot(df[column], kde=has_kde, color=(230 / 255, 111 / 255, 81 / 255), line_kws={'linewidth': 5})
            plt.xlabel('$\Delta v$ ($m/s$)', fontsize=50)
        else:
            plt.title(f'Distribution of $v$', fontsize=50)
            sns.histplot(df[column], kde=has_kde, color=(75 / 255, 101 / 255, 175 / 255), line_kws={'linewidth': 5})
            plt.xlabel('$v$ ($m/s$)', fontsize=50)
        plt.ylabel('Frequency', fontsize=50)
        plt.savefig(output_path + "_" + column + '.png')
        plt.close()


def merge_statistics_results(output_path):
    name = ['Vanderbilt_two_vehicle_ACC',
            'MicroSimACC',
            'CATS_ACC',
            'CATS_platoon',
            'CATS_UW',
            'OpenACC_Casale',
            'OpenACC_Vicolungo',
            'OpenACC_ASta',
            'OpenACC_ZalaZone',
            'Ohio_single_vehicle',
            'Ohio_two_vehicle',
            'Waymo_perception',
            'Waymo_motion',
            'Argoverse2']
    input_path = [[
        './Dataset/Vanderbilt/output/',
        './Dataset/MicroSimACC/output/',
        './Dataset/CATS/output/',
        './Dataset/CATS/output/',
        './Dataset/CATS/output/',
        './Dataset/OpenACC/output/',
        './Dataset/OpenACC/output/',
        './Dataset/OpenACC/output/',
        './Dataset/OpenACC/output/',
        './Dataset/Ohio/output/',
        './Dataset/Ohio/output/',
        './Dataset/Waymo/output/',
        './Dataset/Waymo/output/',
        './Dataset/Argoverse/output/'
    ], [
        '_analysis_two_vehicle_ACC_statistics.csv',
        '_analysis_statistics.csv',
        '_analysis_ACC_statistics.csv',
        '_analysis_platoon_statistics.csv',
        '_analysis_UWM_statistics.csv',
        '_analysis_Casale_statistics.csv',
        '_analysis_Vicolungo_statistics.csv',
        '_analysis_ASta_statistics.csv',
        '_analysis_ZalaZone_statistics.csv',
        '_analysis_single_vehicle_statistics.csv',
        '_analysis_two_vehicle_statistics.csv',
        '_analysis_perception_statistics.csv',
        '_analysis_motion_statistics.csv',
        '_analysis_statistics.csv'
    ]]

    for step in ['step1', 'step2', 'step3']:

        temp_df = pd.DataFrame()
        for i in range(len(input_path[0])):
            file_path = input_path[0][i] + step + input_path[1][i]
            df = pd.read_csv(file_path, index_col=0)

            print(f"{name[i]}, {step} , {df.loc['count', 'Speed_LV']}")

            rows_to_keep = ['mean', 'std', 'min', 'max']
            df = df.loc[df.index.isin(rows_to_keep)]

            df = df.reset_index().melt(id_vars=['index'], var_name='variable', value_name='value')

            df.columns = ['Statistics', 'Variables', name[i]]

            if temp_df.empty:
                temp_df = df
            else:
                temp_df = pd.concat([temp_df, df])

        temp_df = temp_df.groupby(['Variables', 'Statistics'], sort=False).sum().reset_index()
        temp_df.to_csv(output_path + step + '.csv', index=False)


def analyze_AV_performance(input_path, output_path):
    df = pd.read_csv(input_path)

    # Define a function to calculate Time to Collision (TTC), which measures safety.
    def calculate_TTC(space_gap, speed_diff):
        if speed_diff >= 0:
            return np.nan  # If relative speed is non-negative, TTC is not defined (no collision expected).
        return -space_gap / speed_diff

    # Define a matrix for calculating the VT model, a vehicular dynamics model.
    K_matrix = np.array([
        [-7.537, 0.4438, 0.1716, -0.0420],
        [0.0973, 0.0518, 0.0029, -0.0071],
        [-0.003, -7.42e-04, 1.09e-04, 1.16e-04],
        [5.3e-05, 6e-06, -1e-05, -6e-06]
    ])

    # Calculate the VT model for fuel consumption and environmental impact.
    def calculate_VT_model(v, a, K):
        sum_j1_j2 = 0
        for j1 in range(4):
            for j2 in range(4):
                sum_j1_j2 += K[j1][j2] * (v ** j1) * (a ** j2)
        F = np.exp(sum_j1_j2)
        return F

    # Define the Vehicle Specific Power (VSP) model for vehicular dynamics.
    def calculate_VSP(v, a):
        return v * (1.1 * a + 0.132) + 3.02 * 10 ** (-4) * v ** 3

    def calculate_VSP_model(v, a):
        VSP = calculate_VSP(v, a)
        if VSP < -10:
            return 2.48e-03
        elif -10 <= VSP < 10:
            return 1.98e-03 * VSP ** 2 + 3.97e-02 * VSP + 2.01e-01
        else:
            return 7.93e-02 * VSP + 2.48e-03

    # Define the ARRB model, another vehicular dynamics model.
    def calculate_ARRB_model(v, a):
        return (0.666 + 0.019 * v + 0.001 * v ** 2 + 0.0005 * v ** 3 + 0.122 * a + 0.793 * max(a, 0) ** 2)

    # Process each group of data by calculating TTC.
    df['TTC'] = df.apply(lambda row: calculate_TTC(row['Spatial_Gap'], row['Speed_Diff']), axis=1)

    df['Time_Headway'] = df['Spatial_Headway'] / df['Speed_FAV']

    df['Acc_speed_squared_deviation'] = (df['Acc_FAV'] - df['Acc_FAV'].mean()) ** 2 / df['Speed_FAV'].mean()
    df['Speed_squared_deviation'] = (df['Speed_FAV'] - df['Speed_FAV'].mean()) ** 2
    df['Acc_squared_deviation'] = (df['Acc_FAV'] - df['Acc_FAV'].mean()) ** 2

    df['VT_micro_model'] = df.apply(
        lambda row: calculate_VT_model(row['Speed_FAV'], row['Acc_FAV'], K_matrix), axis=1)

    df['VSP_model'] = df.apply(lambda row: calculate_VSP_model(row['Speed_FAV'], row['Acc_FAV']),
                               axis=1) / 800

    df['ARRB_model'] = df.apply(lambda row: calculate_ARRB_model(row['Speed_FAV'], row['Acc_FAV']),
                                axis=1) / 1000

    df['Fuel_consumption'] = df[['VT_micro_model', 'VSP_model', 'ARRB_model']].mean(axis=1)

    df = df[['TTC', 'Time_Headway', 'Acc_squared_deviation', 'Acc_speed_squared_deviation',
             'Speed_squared_deviation', 'Fuel_consumption', 'VT_micro_model', 'VSP_model', 'ARRB_model']]
    df.to_csv(output_path, index=False)


def draw_scatter(input_path, output_path):
    df = pd.read_csv(input_path)

    df['Smoothed_Acc_FAV'] = df['Acc_FAV'].rolling(window=3).mean()
    df.loc[2:, 'Acc_FAV'] = df.loc[2:, 'Smoothed_Acc_FAV']

    colors = [(42 / 255, 157 / 255, 140 / 255),
              (233 / 255, 196 / 255, 107 / 255),
              (230 / 255, 111 / 255, 81 / 255)]

    plt.figure(figsize=(8, 8))  # Set the size of the image here
    plt.scatter(df['Spatial_Gap'], df['Acc_FAV'], color=colors[0], s=2)
    # plt.title('Relationship of $a^{\mathrm{f}}$ and $d$', fontsize=40)
    # plt.xlabel('$d$ ($\mathrm{m}$)', fontsize=30)
    # plt.ylabel('$a^{\mathrm{f}}$ ($\mathrm{m}/\mathrm{s}^2$)', fontsize=30)
    # plt.tick_params(axis='x', labelsize=30)
    # plt.tick_params(axis='y', labelsize=30)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path + f'_Spatial_Gap.png')  # _{name}
    plt.close()

    plt.figure(figsize=(8, 8))  # Set the size of the image here
    plt.scatter(df['Speed_FAV'], df['Acc_FAV'], color=colors[1], s=2)
    # plt.title('Relationship of $a^{\mathrm{f}}$ and $v^{\mathrm{f}}$', fontsize=40)
    # plt.xlabel('$v^{\mathrm{f}}$ ($\mathrm{m}/\mathrm{s}$)', fontsize=30)
    # plt.ylabel('$a^{\mathrm{f}}$ ($\mathrm{m}/\mathrm{s}^2$)', fontsize=30)
    # plt.tick_params(axis='x', labelsize=30)
    # plt.tick_params(axis='y', labelsize=30)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path + f'_Speed_FAV.png')  # _{name}
    plt.close()

    plt.figure(figsize=(8, 8))  # Set the size of the image here
    plt.scatter(df['Speed_Diff'], df['Acc_FAV'], color=colors[2], s=2)
    # plt.title('Relationship of $a^{\mathrm{f}}$ and $\Delta v$', fontsize=40)
    # plt.xlabel('$\Delta v$ ($\mathrm{m}/\mathrm{s}$)', fontsize=30)
    # plt.ylabel('$a^{\mathrm{f}}$ ($\mathrm{m}/\mathrm{s}^2$)', fontsize=30)
    # plt.tick_params(axis='x', labelsize=30)
    # plt.tick_params(axis='y', labelsize=30)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path + f'_Speed_Diff.png')  # _{name}
    plt.close()



def draw_2D_perfromance_metrics(input_paths, output_path, dataset_labels):
    columns = ['TTC', 'Time_Headway', 'Acc_squared_deviation', 'Fuel_consumption',]#

    colors = [
        (250 / 251, 134 / 255, 0),
        '#54B345',
        '#05B9E2',
        (231 / 255, 56 / 255, 71 / 255),
        (131 / 255, 64 / 255, 38 / 255)
    ]

    dfs = [pd.read_csv(path) for path in input_paths]

    for col in columns:
        plt.figure(figsize=(10, 8))  # Set the size of the image here

        bins = None
        if col == 'TTC':
            plt.title(f'Distribution of $TTC$', fontsize=40)
            bins = np.linspace(0, 250, 100)
            x_label = '$TTC$ ($\mathrm{s}$)'

        if col == 'Time_Headway':
            plt.title(r'Distribution of $\tau$', fontsize=40)
            bins = np.linspace(0, 8, 500)
            x_label = r'$\tau$ ($\mathrm{s}$)'

        if col == 'Acc_squared_deviation':
            plt.title(r'Distribution of $\alpha$', fontsize=40)
            bins = np.linspace(0, 0.4, 500)
            x_label = r'$\alpha$ ($\mathrm{m}^2/\mathrm{s}^4$)'

        if col == 'Fuel_consumption':
            plt.title(f'Distribution of $F$', fontsize=40)
            bins = np.linspace(0, 0.01, 500)
            x_label = '$F$ ($\mathrm{L}/\mathrm{s}$)'

        for i, df in enumerate(dfs):
            data = df[col].dropna()
            hist, edges = np.histogram(data, bins=bins, density=True)  #
            x = (edges[:-1] + edges[1:]) / 2  # Compute the centers of histogram bins

            repeated_x = np.repeat(x, (hist * 1000).astype(int))  # Replicate data points based on their weights

            # Plot KDE using replicated data points
            sns.kdeplot(repeated_x, fill=True, label=dataset_labels[i], color=colors[len(colors) - i - 1],
                        alpha=0.4 - 0.05 * i, linewidth=2)

        plt.xlabel(x_label, fontsize=35)
        plt.ylabel('Density', fontsize=35)
        plt.tick_params(axis='both', which='major', labelsize=30)
        plt.xlim(0, None)
        if col == 'Fuel_consumption':
            plt.ylim(None, 3000)
        plt.tight_layout()

        plt.savefig(output_path + col + '.png')
        plt.close()


def draw_2D_labels_statistics(input_paths, output_path, dataset_labels,
                              columns=['Speed_FAV', 'Acc_FAV', 'Spatial_Gap', 'Speed_Diff']):
    colors = [
        (250 / 251, 134 / 255, 0),
        '#54B345',
        '#05B9E2',
        (231 / 255, 56 / 255, 71 / 255),
        (131 / 255, 64 / 255, 38 / 255)
    ]
    dfs = [pd.read_csv(path) for path in input_paths]

    for col in columns:
        plt.figure(figsize=(10, 8))  # Set the size of the image here

        if col == 'Speed_FAV':
            plt.title('Distribution of $v^{\mathrm{f}}$', fontsize=40)
            x_label = '$v^{\mathrm{f}}$ $(\mathrm{m}/\mathrm{s})$'

        if col == 'Acc_FAV':
            plt.title('Distribution of $a^{\mathrm{f}}$', fontsize=40)
            x_label = '$a^\mathrm{f}$ $(\mathrm{m}/\mathrm{s}^2)$'

        if col == 'Spatial_Gap':
            plt.title(f'Distribution of $g$', fontsize=40)
            x_label = '$g$ $(\mathrm{m})$'

        if col == 'Speed_Diff':
            plt.title(f'Distribution of $\Delta v$', fontsize=40)
            x_label = '$\Delta v$ $(\mathrm{m}/\mathrm{s})$'

        for i, df in enumerate(dfs):
            data = df[col].dropna()
            hist, edges = np.histogram(data, density=True)
            x = (edges[:-1] + edges[1:]) / 2  # Compute the centers of histogram bins

            repeated_x = np.repeat(x, (hist * 1000).astype(int))  # Replicate data points based on their weights

            # Plot KDE using replicated data points
            sns.kdeplot(repeated_x, fill=True, label=dataset_labels[i], color=colors[len(colors) - i - 1],
                        alpha=0.5 - 0.07 * i, linewidth=2)

        plt.xlabel(x_label, fontsize=35)
        plt.ylabel('Density', fontsize=35)
        plt.tick_params(axis='both', which='major', labelsize=30)
        # plt.legend(loc='upper right', fontsize=30)

        if col == 'Speed_FAV' or col == 'Spatial_Gap':
            plt.xlim(0, None)

        plt.tight_layout()
        plt.savefig(output_path + col + '.png')
        plt.close()


def draw_statistics_distribution(output_path):
    def crop_center(img):
        img_width, img_height = img.size
        return img.crop((100,
                         160,
                         img_width - 160,
                         img_height - 90))

    img_paths = [
        './Dataset/Vanderbilt/output/step3_analysis_two_vehicle_ACC_',
        './Dataset/MicroSimACC/output/step3_analysis_',
        './Dataset/CATS/output/step3_analysis_ACC_',
        './Dataset/CATS/output/step3_analysis_platoon_',
        './Dataset/CATS/output/step3_analysis_UWM_',
        './Dataset/OpenACC/output/step3_analysis_Casale_',
        './Dataset/OpenACC/output/step3_analysis_Vicolungo_',
        './Dataset/OpenACC/output/step3_analysis_ASta_',
        './Dataset/OpenACC/output/step3_analysis_ZalaZone_',
        './Dataset/Ohio/output/step3_analysis_single_vehicle_',
        './Dataset/Ohio/output/step3_analysis_two_vehicle_',
        './Dataset/Waymo/output/step3_analysis_perception_',
        './Dataset/Waymo/output/step3_analysis_motion_',
        './Dataset/Argoverse/output/step3_analysis_'
    ]

    row_images = []

    for i in range(len(img_paths)):
        column_images = []
        for var in ['Acc_FAV', 'Spatial_Gap', 'Speed_FAV', 'Speed_Diff']:
            column_images.append(Image.open(img_paths[i] + var + '.png'))

        column_images = [crop_center(img) for img in column_images]

        total_height = sum(img.height for img in column_images)
        max_width = max(img.width for img in column_images)

        combined_image_vertical = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for img in column_images:
            combined_image_vertical.paste(img, (0, y_offset))
            y_offset += img.height

        row_images.append(combined_image_vertical)

    total_width = sum(img.width for img in row_images)
    max_height = max(img.height for img in row_images)

    combined_image_horizontal = Image.new('RGB', (total_width, max_height))

    # Paste the images into the new image
    x_offset = 0
    for img in row_images:
        combined_image_horizontal.paste(img, (x_offset, 0))
        x_offset += img.width

    combined_image_horizontal.save(output_path)


def correlation(output_path):
    input_paths = [
        './Dataset/Vanderbilt/output/step3_two_vehicle_ACC.csv',
        './Dataset/MicroSimACC/output/step3.csv',
        './Dataset/CATS/output/step3_ACC.csv',
        './Dataset/CATS/output/step3_platoon.csv',
        './Dataset/CATS/output/step3_UWM.csv',
        './Dataset/OpenACC/output/step3_Casale.csv',
        './Dataset/OpenACC/output/step3_Vicolungo.csv',
        './Dataset/OpenACC/output/step3_ASta.csv',
        './Dataset/OpenACC/output/step3_ZalaZone.csv',
        './Dataset/Ohio/output/step3_single_vehicle.csv',
        './Dataset/Ohio/output/step3_two_vehicle.csv',
        './Dataset/Waymo/output/step3_perception.csv',
        './Dataset/Waymo/output/step3_motion.csv',
        './Dataset/Argoverse/output/step3.csv'
    ]

    all_data = []

    for i, path in enumerate(input_paths):
        df = pd.read_csv(path)
        pearson_corr = df[['Acc_FAV', 'Spatial_Gap', 'Speed_FAV', 'Speed_Diff']].corr()
        spearman_corr = df[['Acc_FAV', 'Spatial_Gap', 'Speed_FAV', 'Speed_Diff']].corr(method='spearman')
        corr_data = {
            'ID': i + 1,
            'Pearson_Spatial_Gap': pearson_corr.at['Acc_FAV', 'Spatial_Gap'],
            'Spearman_Spatial_Gap': spearman_corr.at['Acc_FAV', 'Spatial_Gap'],
            'Pearson_Speed_FAV': pearson_corr.at['Acc_FAV', 'Speed_FAV'],
            'Spearman_Speed_FAV': spearman_corr.at['Acc_FAV', 'Speed_FAV'],
            'Pearson_Speed_Diff': pearson_corr.at['Acc_FAV', 'Speed_Diff'],
            'Spearman_Speed_Diff': spearman_corr.at['Acc_FAV', 'Speed_Diff']
        }
        all_data.append(corr_data)

    combined_data = pd.DataFrame(all_data)
    combined_data.to_csv(output_path, index=False)

def check_row_number():
    paths = [
        './Dataset/Vanderbilt/output/step2_two_vehicle_ACC.csv',
        './Dataset/CATS/output/step2_ACC.csv',
        './Dataset/CATS/output/step2_platoon.csv',
        './Dataset/CATS/output/step2_UWM.csv',
        './Dataset/OpenACC/output/step2_Casale.csv',
        './Dataset/OpenACC/output/step2_Vicolungo.csv',
        './Dataset/OpenACC/output/step2_ASta.csv',
        './Dataset/OpenACC/output/step2_ZalaZone.csv',
        './Dataset/Ohio/output/step2_single_vehicle.csv',
        './Dataset/Ohio/output/step2_two_vehicle.csv',
        './Dataset/Waymo/output/step2_perception.csv',
        './Dataset/Waymo/output/step2_motion.csv',
        './Dataset/Argoverse/output/step2.csv'
    ]

    total_rows = 0

    for path in paths:
        try:
            df = pd.read_csv(path)
            total_rows += len(df)
        except Exception as e:
            print(f"Error reading {path}: {e}")

    print(f"Total rows across all CSV files: {total_rows}")