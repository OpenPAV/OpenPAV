import pandas as pd
import matplotlib as mpl
import numpy as np

def fill_and_clean(input_path, linear_fill_in, outlier,
                   space_gap_upper, space_gap_lower, speed_FAV_upper, speed_FAV_lower,
                   speed_LV_upper, speed_LV_lower, acc_FAV_upper, acc_FAV_lower,
                   acc_LV_upper, acc_LV_lower):
    df = pd.read_csv(input_path)

    # Remove non-following mode data based on defined boundaries for each variable.
    df.loc[~(df['Spatial_Gap'].between(space_gap_lower, space_gap_upper)), 'Spatial_Gap'] = np.nan
    df.loc[~(df['Speed_FAV'].between(speed_FAV_lower, speed_FAV_upper)), 'Speed_FAV'] = np.nan
    df.loc[~(df['Acc_FAV'].between(acc_FAV_lower, acc_FAV_upper)), 'Acc_FAV'] = np.nan
    df.loc[~(df['Speed_LV'].between(speed_LV_lower, speed_LV_upper)), 'Speed_LV'] = np.nan
    df.loc[~(df['Acc_LV'].between(acc_LV_lower, acc_LV_upper)), 'Acc_LV'] = np.nan

    # Replace infinities with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    rows_to_delete = set()
    columns = ['Speed_FAV', 'Acc_FAV', 'Speed_LV', 'Acc_LV', 'Spatial_Gap', 'Speed_Diff']
    for i in range(len(columns)):
        column = columns[i]
        # Remove values beyond ±3 standard deviations iteratively until no changes are needed.
        if outlier is not None and outlier[i] is not None:
            while True:
                mean = df[column].mean(skipna=True)
                std = df[column].std(skipna=True)
                outliers_condition = (df[column] < mean - outlier[i] * std) | (df[column] > mean + outlier[i] * std)
                if not df.loc[outliers_condition, column].empty:
                    df.loc[outliers_condition, column] = np.nan
                else:
                    break

        # Delete groups with excessive missing data.
        if linear_fill_in > 0:
            is_na = df[column].isna()
            na_groups = is_na.ne(is_na.shift()).cumsum()
            na_groups_sizes = na_groups[is_na].value_counts()
            groups_to_delete = na_groups_sizes[na_groups_sizes > linear_fill_in].index
            rows_to_delete.update(na_groups[na_groups.isin(groups_to_delete)].index)

    # Remove rows identified in the previous step
    df.drop(index=rows_to_delete, inplace=True)
    # Perform linear interpolation within each trajectory group
    df.groupby('Trajectory_ID').apply(lambda group: group.interpolate(method='linear')).reset_index(drop=True)
    # Drop rows that still contain NaNs after interpolation
    df.dropna(inplace=True)

    return df

def revise_traj_id(df, output_path, time_step, fill_row_num, fill_start, fill_end, update_time=True):
    # Filter out trajectories shorter than a specified length.
    df = df.groupby('Trajectory_ID').filter(lambda x: len(x) >= fill_row_num)

    # Update time indices to ensure continuity if specified.
    if update_time:
        current_traj_ID = 0
        current_time_ID = 0
        df['Trajectory_ID'] = current_traj_ID
        df['new_Time_Index'] = current_time_ID

        previous_time_ID = df.iloc[0]['Time_Index'] - time_step
        for index, row in df.iterrows():
            if index > 0 and abs(row['Time_Index'] - previous_time_ID) > time_step + 1e-5:
                current_traj_ID += 1
                current_time_ID = 0
            else:
                current_time_ID += time_step

            df.at[index, 'Trajectory_ID'] = current_traj_ID
            df.at[index, 'new_Time_Index'] = current_time_ID
            previous_time_ID = row['Time_Index']

        df['Time_Index'] = df['new_Time_Index']
        df.drop(columns=['new_Time_Index'], inplace=True)

    # Again filter trajectories that are too short.
    df = df.groupby('Trajectory_ID').filter(lambda x: len(x) >= fill_row_num)

    # Remove unstable trajectory sections.
    indices_to_remove = []
    for Trajectory_ID, group in df.groupby('Trajectory_ID'):
        indices_max = group.nlargest(fill_end, 'Time_Index').index
        indices_min = group.nsmallest(fill_start, 'Time_Index').index
        indices_to_remove.extend(indices_max)
        indices_to_remove.extend(indices_min)
    df.drop(indices_to_remove, inplace=True)
    df = df.reset_index(drop=True)
    df['Time_Index'] -= fill_start * time_step

    # Adjust positions relative to the start of each trajectory.
    def adjust_positions(group):
        first_Pos_FAV = group['Pos_FAV'].iloc[0]
        group['Pos_FAV'] -= first_Pos_FAV
        group['Pos_LV'] -= first_Pos_FAV
        return group

    df = df.groupby('Trajectory_ID').apply(adjust_positions)

    # Update trajectory IDs to ensure continuity.
    unique_Trajectory_IDs = df['Trajectory_ID'].unique()
    Trajectory_ID_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_Trajectory_IDs)}
    df['Trajectory_ID'] = df['Trajectory_ID'].map(Trajectory_ID_mapping)

    df.to_csv(output_path, index=False)

def merge_data(merge_data_list, output_path):
    df_list = []
    max_value = 0
    for path in merge_data_list:
        df = pd.read_csv(path)
        if not df.empty:
            df['Trajectory_ID'] += max_value
            max_value = df['Trajectory_ID'].max() + 1
            df_list.append(df)
    merged = pd.concat(df_list)
    merged.to_csv(output_path, index=False)
