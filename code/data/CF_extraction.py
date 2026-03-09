import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from data_transformation import *
from data_cleaning import *


def Waymo_extract_df(input_path):
    state_features = {
        'state/past/x':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/y':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/speed':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/past/length':
            tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
        'state/current/x':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/y':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/speed':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/current/length':
            tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
        'state/future/x':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/y':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/speed':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/future/length':
            tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
        'state/id':
            tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        'state/type':
            tf.io.FixedLenFeature([128], tf.float32, default_value=None),
        'state/is_sdc':
            tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, state_features)

    raw_dataset = tf.data.TFRecordDataset(input_path)
    parsed_dataset = raw_dataset.map(_parse_function)

    data_rows = []

    traj_id = 0
    for parsed_record in parsed_dataset:
        def extract_data(field_name):
            return parsed_record[field_name].numpy()

        data_to_extract = ['state/current/x', 'state/current/y', 'state/current/speed', 'state/current/length',
                           'state/past/x', 'state/past/y', 'state/past/speed', 'state/past/length',
                           'state/future/x', 'state/future/y', 'state/future/speed', 'state/future/length',
                           'state/id', 'state/is_sdc', 'state/type']

        extracted_data = {key: extract_data(key) for key in data_to_extract}

        def add_trajectory_data(time_range, data, time):
            for j in time_range:
                row = {'Trajectory_ID': traj_id}
                for i in range(128):
                    row.update({
                        f'id_{i}': data['state/id'][i],
                        f'is_av_{i}': data['state/is_sdc'][i],
                        f'type_{i}': data['state/type'][i],
                        f'x_{i}': data['state/' + time + '/x'][i][j],
                        f'y_{i}': data['state/' + time + '/y'][i][j],
                        f'length_{i}': data['state/' + time + '/length'][i][j],
                        f'speed_{i}': data['state/' + time + '/speed'][i][j],
                    })
                data_rows.append(row)

        add_trajectory_data(range(10), extracted_data, 'past')
        add_trajectory_data(range(1), extracted_data, 'current')
        add_trajectory_data(range(80), extracted_data, 'future')

        traj_id += 1

    dataframe = pd.DataFrame(data_rows)
    return dataframe


def Waymo_extract_cf_traj(data, straight_threshold=0.9, direction_threshold=0.985,
                          relative_diff_threshold=0.2):
    """
    Processes trajectories to filter out non-straight or non-following ones based on specified thresholds.

    Parameters:
    - data: DataFrame containing trajectory data.
    - straight_threshold: R² threshold to determine if the trajectory is straight.
    - stable_threshold: Unused parameter in this context.
    - direction_threshold: Cosine similarity threshold to determine if trajectories are moving in the same direction.
    - relative_diff_threshold: Threshold to filter out trajectories based on relative difference in calculated values.
    """
    # 1. Remove non-straight trajectories
    is_traj_stright = {}
    av_directions = {}
    for traj_id, group in data.groupby('Trajectory_ID'):
        # Step 1: Identify if all 'is_av_{i}' columns are 1 for each trajectory
        av_idx = None
        for i in range(128):
            if all(group[f'is_av_{i}'] == 1):
                av_idx = i
                break
        if av_idx is None:
            continue

        # Extract 'x' and 'y' coordinates and fit a linear regression model
        X = group[[f'x_{av_idx}']].values.reshape(-1, 1)
        y = group[f'y_{av_idx}'].values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        # Check if the trajectory is straight based on the R² value
        if r2 >= straight_threshold:
            slope = model.coef_[0][0]
            direction_vector = np.array([1, slope])
            direction_norm = np.linalg.norm(direction_vector)
            direction_vector = direction_vector / direction_norm
            av_directions[traj_id] = (av_idx, direction_vector)
            is_traj_stright[traj_id] = True
        else:
            is_traj_stright[traj_id] = False

    # Remove trajectories that are not straight
    traj_ids_to_delete = [traj_id for traj_id, result in is_traj_stright.items() if result == False]
    data = data[~data['Trajectory_ID'].isin(traj_ids_to_delete)]

    # 2. Remove trajectories that are not following another vehicle
    nearest_vehicle_indices = {}
    traj_ids_to_delete = []
    cosine_threshold = direction_threshold

    for traj_id, group in data.groupby('Trajectory_ID'):
        av_idx = av_directions[traj_id][0]
        ref_x_direction, ref_y_direction = av_directions[traj_id][1]
        nearest_vehicles_per_row = []
        found_nearest_in_all_rows = True
        first_row = True
        for row_idx, row in group.iterrows():
            min_distance = float('inf')
            nearest_vehicle_idx = None
            for i in range(128):
                if i == av_idx:
                    continue

                # Skip opposite direction traffic
                if not first_row:
                    first_row = False
                    current_row = row
                    previous_row = group.loc[row_idx - 1]
                    dot_product = (current_row[f'x_{i}'] - previous_row[f'x_{i}']) * \
                                  (current_row[f'x_{av_idx}'] - previous_row[f'x_{av_idx}']) + \
                                  (current_row[f'y_{i}'] - previous_row[f'y_{i}']) * \
                                  (current_row[f'y_{av_idx}'] - previous_row[f'y_{av_idx}'])
                    if dot_product < 0:
                        continue

                # Remove vehicles not in a straight line
                x_direction = row[f'x_{i}'] - row[f'x_{av_idx}']
                y_direction = row[f'y_{i}'] - row[f'y_{av_idx}']
                vector_length = (x_direction ** 2 + y_direction ** 2) ** 0.5
                vector_cosine = ((x_direction * ref_x_direction + y_direction * ref_y_direction)
                                 / vector_length) if vector_length != 0 else 0
                # Check if moving in the same direction based on cosine similarity
                if vector_cosine >= cosine_threshold:
                    distance = vector_length
                    if distance < min_distance:
                        min_distance = distance
                        nearest_vehicle_idx = i
            if nearest_vehicle_idx is None:
                found_nearest_in_all_rows = False
                break
            else:
                nearest_vehicles_per_row.append(nearest_vehicle_idx)

        # Add trajectories to deletion list if they don't consistently follow the same vehicle
        if not found_nearest_in_all_rows:
            traj_ids_to_delete.append(traj_id)
        elif len(set(nearest_vehicles_per_row)) > 1:
            traj_ids_to_delete.append(traj_id)
        else:
            nearest_vehicle_indices[traj_id] = nearest_vehicles_per_row[0]

    # Delete trajectories that do not follow a single vehicle
    data = data[~data['Trajectory_ID'].isin(traj_ids_to_delete)]

    # 3. Organize a new DataFrame with filtered data
    new_rows = []
    for traj_id, group in data.groupby('Trajectory_ID'):
        if traj_id not in av_directions or traj_id not in nearest_vehicle_indices:
            continue

        av_idx = av_directions[traj_id][0]
        nearest_idx = nearest_vehicle_indices[traj_id]

        if nearest_idx is None:
            continue
        for _, row in group.iterrows():
            follower_x = row[f'x_{av_idx}']
            follower_y = row[f'y_{av_idx}']
            follower_length = row[f'length_{av_idx}']
            follower_speed = row[f'speed_{av_idx}']
            leader_x = row[f'x_{nearest_idx}']
            leader_y = row[f'y_{nearest_idx}']
            leader_length = row[f'length_{nearest_idx}']
            leader_speed = row[f'speed_{nearest_idx}']
            new_row = {'Trajectory_ID': traj_id, 'leader_speed': leader_speed, 'follower_speed': follower_speed,
                       'leader_x': leader_x, 'leader_y': leader_y, 'leader_length': leader_length,
                       'follower_x': follower_x, 'follower_y': follower_y, 'follower_length': follower_length}
            new_rows.append(new_row)

    df = pd.DataFrame(new_rows)

    # 4. Remove trajectories where the space gap and speed difference do not match
    num_before = len(df)

    df['Space_Gap'] = (np.sqrt(
        (df['leader_x'] - df['follower_x']) ** 2 + (df['leader_y'] - df['follower_y']) ** 2)) - df[
                          'leader_length'] / 2 - df['follower_length'] / 2

    df['Speed_Diff'] = df['leader_speed'] - df['follower_speed']

    grouped = df.groupby('Trajectory_ID').agg(
        Speed_Diff_Mean=('Speed_Diff', 'mean'),
        Space_Gap_Change=('Space_Gap', lambda x: x.iloc[-1] - x.iloc[0])
    )

    grouped['Relative_Diff'] = abs(grouped['Space_Gap_Change'] - (grouped['Speed_Diff_Mean'] * 9.1)) / (
            grouped['Speed_Diff_Mean'] * 9.1)

    traj_to_remove_relative = grouped[grouped['Relative_Diff'] > relative_diff_threshold].index

    df = df[~df['Trajectory_ID'].isin(traj_to_remove_relative)]

    df.drop(['Space_Gap', 'Speed_Diff'], axis=1, inplace=True)

    num_after = len(df)
    print(f'total traj num: {num_before / 91}, delete traj num: {(num_before - num_after) / 91}.')

    return df


def Argo2_extract_df(input_paths, traj_id):
    df_list = []
    for input_path in input_paths:
        data = pd.read_parquet(input_path)

        df = pd.DataFrame(
            columns=['Trajectory_ID', 'Time_Index', 'ID', 'x', 'y', 'speed'])
        df['Time_Index'] = data.iloc[:, 4]
        df['Trajectory_ID'] = traj_id
        df['ID'] = data.iloc[:, 1].replace({'AV': 0})
        df['x'] = data.iloc[:, 5]
        df['y'] = data.iloc[:, 6]
        df['speed'] = np.linalg.norm(data.iloc[:, [8, 9]].values, axis=1)

        def reshape_group_optimized(group):
            group = group.reset_index(drop=True)
            reshaped_data_info = group.loc[0, ['Trajectory_ID', 'Time_Index']].to_dict()
            columns_data = []
            for i, row in group.iterrows():
                suffix = f"_{row['ID']}"
                temp_df = pd.DataFrame({
                    f'ID_{suffix}': [row['ID']],
                    f'x_{suffix}': [row['x']],
                    f'y_{suffix}': [row['y']],
                    f'speed_{suffix}': [row['speed']]
                })
                columns_data.append(temp_df)
            reshaped_data = pd.concat([pd.DataFrame(reshaped_data_info, index=[0])] + columns_data, axis=1)
            return reshaped_data

        df = df.groupby(['Trajectory_ID', 'Time_Index']).apply(reshape_group_optimized).reset_index(
            drop=True)
        df = df.groupby(['Trajectory_ID', 'Time_Index']).first().unstack(
            fill_value=0).stack(future_stack=True).reset_index()

        columns = ['Trajectory_ID', 'Time_Index']
        for i in range(int((len(df.columns) - 2) / 4)):
            columns += [f'ID_{i}', f'x_{i}', f'y_{i}', f'speed_{i}']
        df.columns = columns

        df_list.append(df)

        traj_id += 1

    df = pd.concat(df_list)
    return df


def Argo2_extract_cf_traj(data, output_path, straight_threshold=0.9, direction_threshold=0.985,
                          relative_diff_threshold=0.2):
    """
    Filters and processes car-following trajectories from the Argoverse dataset.

    Parameters:
    - data: DataFrame containing trajectory data.
    - output_path: Path where the processed data should be saved.
    - straight_threshold: Threshold for R^2 to consider a trajectory as straight.
    - stable_threshold: Unused threshold, might be reserved for future use.
    - direction_threshold: Cosine similarity threshold to determine direction alignment.
    - relative_diff_threshold: Threshold for the relative difference in speed and gap changes.
    """

    # 1. Remove non-straight trajectories
    is_traj_straight = {}
    av_directions = {}
    for traj_id, group in data.groupby('Trajectory_ID'):
        # Step 1: Identify which vehicle is the AV and check if all its trajectory points are labeled as AV
        av_idx = None
        for i in range(int((len(group.columns) - 2) / 4)):
            if all(group[f'ID_{i}'] == 0):
                av_idx = i
                break
        if av_idx is None:
            continue

        # Extract x and y coordinates and fit a linear regression model
        X = group[[f'x_{av_idx}']].values.reshape(-1, 1)
        y = group[[f'y_{av_idx}']].values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        # Check if the trajectory is straight using the R² value
        if r2 >= straight_threshold:
            slope = model.coef_[0][0]
            direction_vector = np.array([1, slope])
            direction_norm = np.linalg.norm(direction_vector)
            direction_vector /= direction_norm
            av_directions[traj_id] = (av_idx, direction_vector)
            is_traj_straight[traj_id] = True
        else:
            is_traj_straight[traj_id] = False

    # Remove trajectories that are not considered straight
    traj_ids_to_delete = [traj_id for traj_id, result in is_traj_straight.items() if not result]
    data = data[~data['Trajectory_ID'].isin(traj_ids_to_delete)]

    # 2. Remove trajectories that are not following any vehicle
    nearest_vehicle_indices = {}
    traj_ids_to_delete = []
    for traj_id, group in data.groupby('Trajectory_ID'):
        av_idx = av_directions[traj_id][0]
        ref_x_direction, ref_y_direction = av_directions[traj_id][1]
        nearest_vehicles_per_row = []
        found_nearest_in_all_rows = True
        first_row = True
        for row_idx, row in group.iterrows():
            min_distance = float('inf')
            nearest_vehicle_idx = None
            for i in range(int((len(group.columns) - 2) / 4)):
                if i == av_idx:
                    continue

                # Exclude oncoming traffic
                if not first_row:
                    first_row = False
                    current_row = row
                    previous_row = group.loc[row_idx - 1]
                    dot_product = (current_row[f'x_{i}'] - previous_row[f'x_{i}']) * \
                                  (current_row[f'x_{av_idx}'] - previous_row[f'x_{av_idx}']) + \
                                  (current_row[f'y_{i}'] - previous_row[f'y_{i}']) * \
                                  (current_row[f'y_{av_idx}'] - previous_row[f'y_{av_idx}'])
                    if dot_product < 0:
                        continue

                # Check if vehicles are moving in a straight line and in the same direction
                x_direction = row[f'x_{i}'] - row[f'x_{av_idx}']
                y_direction = row[f'y_{i}'] - row[f'y_{av_idx}']
                vector_length = np.sqrt(x_direction ** 2 + y_direction ** 2)
                vector_cosine = ((x_direction * ref_x_direction + y_direction * ref_y_direction) /
                                 vector_length) if vector_length != 0 else 0
                if vector_cosine >= direction_threshold:
                    distance = vector_length
                    if distance < min_distance:
                        min_distance = distance
                        nearest_vehicle_idx = i
            if nearest_vehicle_idx is None:
                found_nearest_in_all_rows = False
                break
            else:
                nearest_vehicles_per_row.append(nearest_vehicle_idx)

        # Remove trajectories if they do not consistently follow the same vehicle
        if not found_nearest_in_all_rows or len(set(nearest_vehicles_per_row)) > 1:
            traj_ids_to_delete.append(traj_id)
        else:
            nearest_vehicle_indices[traj_id] = nearest_vehicles_per_row[0]

    # Remove data for trajectories without a consistent following vehicle
    data = data[~data['Trajectory_ID'].isin(traj_ids_to_delete)]

    # 3. Prepare a new DataFrame with the filtered data
    new_rows = []
    for traj_id, group in data.groupby('Trajectory_ID'):
        if traj_id not in av_directions or traj_id not in nearest_vehicle_indices:
            continue  # Skip if the trajectory ID is not in the filtered set

        av_idx = av_directions[traj_id][0]
        nearest_idx = nearest_vehicle_indices[traj_id]
        for _, row in group.iterrows():
            leader_x = row[f'x_{av_idx}']
            leader_y = row[f'y_{av_idx}']
            leader_speed = row[f'speed_{av_idx}']
            follower_x = row[f'x_{nearest_idx}']
            follower_y = row[f'y_{nearest_idx}']
            follower_speed = row[f'speed_{nearest_idx}']
            new_row = {'Trajectory_ID': traj_id, 'leader_speed': leader_speed, 'follower_speed': follower_speed,
                       'leader_x': leader_x, 'leader_y': leader_y, 'follower_x': follower_x, 'follower_y': follower_y}
            new_rows.append(new_row)

    # Create DataFrame from the filtered data
    df = pd.DataFrame(new_rows)
    df = df.reset_index(drop=True)
    row_count = data.shape[0]
    if row_count % 110 != 0:
        print(f"Error: The number of rows ({row_count}) is not a multiple of 110.")
    df['Trajectory_ID'] = df.index // 110

    # 4. Remove trajectories where the space gap and speed difference do not match
    num_before = len(df)

    df['Space_Gap'] = (np.sqrt(
        (df['leader_x'] - df['follower_x']) ** 2 + (df['leader_y'] - df['follower_y']) ** 2)) - default_vehicle_length

    df['Speed_Diff'] = df['leader_speed'] - df['follower_speed']

    grouped = df.groupby('Trajectory_ID').agg(
        Speed_Diff_Mean=('Speed_Diff', 'mean'),
        Space_Gap_Change=('Space_Gap', lambda x: x.iloc[-1] - x.iloc[0])
    )

    grouped['Relative_Diff'] = abs(grouped['Space_Gap_Change'] - (grouped['Speed_Diff_Mean'] * 11)) / (
            grouped['Speed_Diff_Mean'] * 11)

    traj_to_remove_relative = grouped[grouped['Relative_Diff'] > relative_diff_threshold].index

    df = df[~df['Trajectory_ID'].isin(traj_to_remove_relative)]

    df.drop(['Space_Gap', 'Speed_Diff'], axis=1, inplace=True)

    num_after = len(df)
    print(
        f'Total number of trajectories: {num_before / 110}, number of trajectories deleted: {(num_before - num_after) / 110}.')

    df.to_csv(output_path, index=False)


def combine():
    for i in range(1):
        merge_data_list = []
        for j in range(25):
            cf_path = f'./Dataset/Argoverse/data/val/CF_trajectories_{i * 25 + j + 1}.csv'
            merge_data_list.append(cf_path)
        merge_data_path = f'./Dataset/Argoverse/output/step0_CF_trajectory_{i}.csv'
        merge_data(merge_data_list, merge_data_path)
