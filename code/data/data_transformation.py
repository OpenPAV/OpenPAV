import pandas as pd
from geopy.distance import geodesic
import numpy as np

default_vehicle_length = 4.5


def Vanderbilt_convert_format(input_path, output_path):
    data = pd.read_csv(input_path, header=None)

    data.insert(0, 'Trajectory_ID', 0)
    for idx in range(3):
        data.insert(2 + idx, f'col_1{idx}', 0)
    for idx in range(3):
        data.insert(6 + idx, f'col_2{idx}', 0)
    for idx in range(2):
        data.insert(12 + idx, f'col_3{idx}', 0)

    data.columns = ['Trajectory_ID', 'Time_Index',
                    'ID_LV', 'Type_LV', 'Pos_LV', 'Speed_LV', 'Acc_LV',
                    'ID_FAV', 'Pos_FAV', 'Speed_FAV', 'Acc_FAV',
                    'Spatial_Gap', 'Spatial_Headway', 'Speed_Diff']

    temp = data['Speed_LV'].copy()
    data['Speed_LV'] = data['Speed_FAV']
    data['Speed_FAV'] = temp
    temp = data['Acc_FAV'].copy()
    data['Acc_FAV'] = data['Spatial_Gap']
    data['Spatial_Gap'] = temp

    data['ID_LV'] = -1
    data['Type_LV'] = 0
    data['ID_FAV'] = 0

    data['Speed_Diff'] = data['Speed_LV'] - data['Speed_FAV']
    data['Spatial_Headway'] = data['Spatial_Gap'] + default_vehicle_length

    data['Acc_LV'] = ((data['Speed_LV'] - data['Speed_LV'].shift(1)) / 0.1).shift(-1)

    average_speed = (data['Speed_FAV'] + data['Speed_FAV'].shift(1)) / 2
    data['Pos_FAV'] = (0.1 * average_speed).cumsum()
    data.loc[0, 'Pos_FAV'] = 0
    data['Pos_LV'] = data['Pos_FAV'] + data['Spatial_Headway']
    data = data.iloc[:-1]

    data.to_csv(output_path, index=False)


def MicroSimACC_convert_format(input_path, output_path):
    data = pd.read_csv(input_path)

    data = data.dropna()
    new_column_order = ['timestamps', 'Smoothed_speed1', 'Smoothed_acceleration1', 'Smoothed_speed2',
                        'Smoothed_acceleration2', 'Difference']

    data['Smoothed_speed1']=data['Smoothed_speed1'] * 0.44704 # mph to m/s
    data['Smoothed_speed2'] = data['Smoothed_speed2'] * 0.44704  # mph to m/s

    data = data[new_column_order]
    data = data.reset_index(drop=True)

    data.insert(0, 'Trajectory_ID', 0)
    for idx in range(3):
        data.insert(2 + idx, f'col_1{idx}', 0)
    for idx in range(2):
        data.insert(7 + idx, f'col_2{idx}', 0)
    data.insert(11, f'col_3', 0)
    data.insert(13, f'col_4', 0)

    data.columns = ['Trajectory_ID', 'Time_Index',
                    'ID_LV', 'Type_LV', 'Pos_LV', 'Speed_LV', 'Acc_LV',
                    'ID_FAV', 'Pos_FAV', 'Speed_FAV', 'Acc_FAV',
                    'Spatial_Gap', 'Spatial_Headway', 'Speed_Diff']

    data['Time_Index'] = np.arange(0, len(data) * 0.2, 0.2)[:len(data)]

    data['ID_LV'] = -1
    data['Type_LV'] = 0
    data['ID_FAV'] = 0

    data['Speed_Diff'] = data['Speed_LV'] - data['Speed_FAV']
    data['Spatial_Gap'] = data['Spatial_Headway'] - default_vehicle_length

    data['Acc_LV'] = ((data['Speed_LV'] - data['Speed_LV'].shift(1)) / 0.1).shift(-1)
    data['Acc_FAV'] = ((data['Speed_FAV'] - data['Speed_FAV'].shift(1)) / 0.1).shift(-1)

    average_speed = (data['Speed_FAV'] + data['Speed_FAV'].shift(1)) / 2
    data['Pos_FAV'] = (0.2 * average_speed).cumsum()
    data.loc[0, 'Pos_FAV'] = 0
    data['Pos_LV'] = data['Pos_FAV'] + data['Spatial_Headway']
    data = data.iloc[:-1]

    data.to_csv(output_path, index=False)


def CATSACC_convert_format(input_path, output_path):
    column_names = ['id', 'time', 'lon', 'lat', 'speed']
    all_sheets = pd.read_excel(input_path, sheet_name=None, header=None, names=column_names, engine='openpyxl')

    def process_time_string(time_str):
        return float(time_str[5:])

    earliest_times = []
    for sheet_name, sheet in all_sheets.items():
        earliest_time_str = sheet['time'].iloc[0]
        earliest_time = process_time_string(earliest_time_str)
        earliest_times.append(earliest_time)
    start_time = min(earliest_times)

    df_list = []
    for i in range(2):
        sheet1, sheet2 = list(all_sheets.values())[i:i + 2]

        sheet1 = sheet1.drop(columns=['id'])
        sheet2 = sheet2.drop(columns=['id'])

        df = pd.merge(sheet1, sheet2, on='time')

        def calculate_distance(row):
            location1 = (row['lat_x'], row['lon_x'])
            location2 = (row['lat_y'], row['lon_y'])
            return geodesic(location1, location2).kilometers * 1000

        df['Spatial_Headway'] = df.apply(calculate_distance, axis=1)
        df.drop(columns=['lat_x', 'lat_y', 'lon_x', 'lon_y'], inplace=True)
        df['time'] = df['time'].apply(lambda t: (process_time_string(t) - start_time))

        if i == 0:
            df.insert(1, 'Id_l', -1)
            df.insert(3, 'Id_f', 0)
            df.insert(0, 'Trajectory_ID', 0)
        else:
            df.insert(1, 'Id_l', 0)
            df.insert(3, 'Id_f', 1)
            df.insert(0, 'Trajectory_ID', 1)

        df_list.append(df)
    data = pd.concat(df_list)

    data.insert(3, f'type_l', 0)
    data.insert(4, f'pos_l', 0)
    data.insert(6, f'acc_l', 0)
    data.insert(8, f'pos_f', 0)
    data.insert(10, f'acc', 0)
    data.insert(11, f'space_gap', 0)
    data.insert(13, f's_diff', 0)

    data.columns = ['Trajectory_ID', 'Time_Index',
                    'ID_LV', 'Type_LV', 'Pos_LV', 'Speed_LV', 'Acc_LV',
                    'ID_FAV', 'Pos_FAV', 'Speed_FAV', 'Acc_FAV',
                    'Spatial_Gap', 'Spatial_Headway', 'Speed_Diff']

    data['Type_LV'] = data['ID_LV'].apply(lambda x: 1 if x in [0, 1] else 0)
    data['Speed_Diff'] = data['Speed_LV'] - data['Speed_FAV']
    data['Spatial_Gap'] = data['Spatial_Headway'] - 4.92

    data = data.reset_index(drop=True)

    data['Acc_LV'] = ((data['Speed_LV'] - data['Speed_LV'].shift(1)) / 0.1).shift(-1)
    data['Acc_FAV'] = ((data['Speed_FAV'] - data['Speed_FAV'].shift(1)) / 0.1).shift(-1)

    average_speed = (data['Speed_FAV'] + data['Speed_FAV'].shift(1)) / 2
    data['Pos_FAV'] = (0.1 * average_speed).cumsum()
    data.loc[0, 'Pos_FAV'] = 0
    data['Pos_LV'] = data['Pos_FAV'] + data['Spatial_Headway']
    data = data.iloc[:-1]

    data['Time_Index'] -= data['Time_Index'].min()

    data.to_csv(output_path, index=False)


def CATSPlatoon_convert_format(input_path, output_path):
    def merge_sheets(file_path):
        column_names = ['id', 'time', 'lat', 'lon', 'speed']
        xls = pd.ExcelFile(file_path)
        dfs = []
        for sheet_name in xls.sheet_names:
            df = xls.parse(sheet_name)
            df.columns = column_names
            df.drop(columns=['id'], inplace=True)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    df_leading = merge_sheets(input_path + "/Leading.xlsx")
    df_mid = merge_sheets(input_path + "/Black-Mid.xlsx")
    df_last = merge_sheets(input_path + "/Red-Last.xlsx")

    def process_time_string(time_str):
        return float(time_str[5:])

    earliest_times = []
    for sheet in [df_leading, df_mid, df_last]:
        earliest_time_str = sheet['time'].iloc[0]
        earliest_time = process_time_string(earliest_time_str)
        earliest_times.append(earliest_time)
    start_time = min(earliest_times)

    df_traj1 = pd.merge(df_leading, df_mid, on='time')
    df_traj2 = pd.merge(df_mid, df_last, on='time')

    count = 0
    for df in [df_traj1, df_traj2]:
        def calculate_distance(row):
            location1 = (row['lat_x'], row['lon_x'])
            location2 = (row['lat_y'], row['lon_y'])
            return geodesic(location1, location2).kilometers * 1000

        df['Spatial_Headway'] = df.apply(calculate_distance, axis=1)
        df.drop(columns=['lat_x', 'lat_y', 'lon_x', 'lon_y'], inplace=True)
        df['time'] = df['time'].apply(lambda t: (process_time_string(t) - start_time))

        if count == 0:
            df.insert(1, 'Id_l', -1)
            df.insert(3, 'Id_f', 0)
            df.insert(0, 'Trajectory_ID', 0)
        else:
            df.insert(1, 'Id_l', 0)
            df.insert(3, 'Id_f', 1)
            df.insert(0, 'Trajectory_ID', 1)
        count += 1

    data = pd.concat([df_traj1, df_traj2])
    
    data.insert(3, f'type_l', 0)
    data.insert(4, f'pos_l', 0)
    data.insert(6, f'acc_l', 0)
    data.insert(8, f'pos_f', 0)
    data.insert(10, f'acc', 0)
    data.insert(11, f'space_gap', 0)
    data.insert(13, f's_diff', 0)

    data.columns = ['Trajectory_ID', 'Time_Index',
                    'ID_LV', 'Type_LV', 'Pos_LV', 'Speed_LV', 'Acc_LV',
                    'ID_FAV', 'Pos_FAV', 'Speed_FAV', 'Acc_FAV',
                    'Spatial_Gap', 'Spatial_Headway', 'Speed_Diff']

    data['Type_LV'] = data['ID_LV'].apply(lambda x: 1 if x in [0, 1] else 0)
    data['Speed_Diff'] = data['Speed_LV'] - data['Speed_FAV']
    data['Spatial_Gap'] = data['Spatial_Headway'] - 4.92
    

    data = data.reset_index(drop=True)

    data['Acc_LV'] = ((data['Speed_LV'] - data['Speed_LV'].shift(1)) / 1).shift(-1)
    data['Acc_FAV'] = ((data['Speed_FAV'] - data['Speed_FAV'].shift(1)) / 1).shift(-1)

    average_speed = (data['Speed_FAV'] + data['Speed_FAV'].shift(1)) / 2
    data['Pos_FAV'] = (1 * average_speed).cumsum()
    data.loc[0, 'Pos_FAV'] = 0
    data['Pos_LV'] = data['Pos_FAV'] + data['Spatial_Headway']
    data = data.iloc[:-1]

    data['Time_Index'] -= data['Time_Index'].min()

    data.to_csv(output_path, index=False)


def CATSUW_convert_format(input_path, output_path):
    data = pd.read_csv(input_path)

    new_column_order = ['time', 'leader_p', 'leader_v', 'follower_p', 'follower_v']
    data = data[new_column_order]

    data.insert(0, 'Trajectory_ID', 0)
    data.insert(2, 'ID_LV', -1)
    data.insert(3, 'Type_LV', -1)
    for idx in range(2):
        data.insert(6 + idx, f'col_2{idx}', 0)
    data.insert(10, f'acc', 0)
    data.insert(11, 'spacing', 0)
    data.insert(12, 'space_headway', 0)
    data.insert(13, 'speed_diff', 0)

    data.columns = ['Trajectory_ID', 'Time_Index',
                    'ID_LV', 'Type_LV', 'Pos_LV', 'Speed_LV', 'Acc_LV',
                    'ID_FAV', 'Pos_FAV', 'Speed_FAV', 'Acc_FAV',
                    'Spatial_Gap', 'Spatial_Headway', 'Speed_Diff']

    data['ID_LV'] = -1
    data['Type_LV'] = 0
    data['ID_FAV'] = 0

    data.to_csv(output_path, index=False)

    data['Spatial_Gap'] = data['Pos_LV'] - data['Pos_FAV']
    data['Speed_Diff'] = data['Speed_LV'] - data['Speed_FAV']
    data['Spatial_Headway'] = data['Spatial_Gap'] + 4.92

    data['Acc_LV'] = ((data['Speed_LV'] - data['Speed_LV'].shift(1)) / 0.1).shift(-1)
    data['Acc_FAV'] = ((data['Speed_FAV'] - data['Speed_FAV'].shift(1)) / 0.1).shift(-1)

    data.to_csv(output_path, index=False)


def OpenACC_convert_format(input_path, output_path, id_map):
    df = pd.read_csv(input_path, header=None, skiprows=1, nrows=1)
    id_row = df.values.tolist()[0][1:]
    id_row = [x for x in id_row if not pd.isna(x)]
    vehicle_ids = list(map(lambda x: id_map.get(x, x), id_row))
    vehicle_num = len(vehicle_ids)

    data = pd.read_csv(input_path, skiprows=5)

    df_list = []
    for i in range(vehicle_num):
        if all(column in data.columns for column in [f'Speed{i}', f'Speed{i + 1}', f'IVS{i}']):
            if f'Driver{i + 1}' in data.columns and f'Driver{i}' in data.columns:
                df = data[['Time', f'Driver{i}', f'Speed{i}', f'Speed{i + 1}', f'IVS{i}', f'Driver{i + 1}']].copy()
                df.columns = ['Time', 'Driver_l', 'Speed_l', 'Speed_f', 'IVS', 'Driver_f']
                df = df[df['Driver_f'] != 'Human']
                df.drop(columns=['Driver_f'], inplace=True)
                df['Driver_l'] = df['Driver_l'].replace({'Human': 0, 'ACC': 1})
                df.insert(1, 'Id_l', vehicle_ids[i - 1])
                df.insert(4, 'Id_f', vehicle_ids[i])
                df.insert(0, 'Trajectory_ID', i - 1)
                df_list.append(df)
            else:
                df = data[['Time', f'Speed{i}', f'Speed{i + 1}', f'IVS{i}']].copy()
                df.columns = ['Time', 'Speed_l', 'Speed_f', 'IVS']
                df.insert(1, 'Id_l', vehicle_ids[i - 1])
                df.insert(2, 'Driver_l', 1)
                df.insert(4, 'Id_f', vehicle_ids[i])
                df.insert(0, 'Trajectory_ID', i - 1)
                df_list.append(df)

    data = pd.concat(df_list)

    data.insert(4, f'pos_l', 0)
    data.insert(6, f'acc_l', 0)
    data.insert(8, f'pos_f', 0)
    data.insert(10, f'acc', 0)
    data.insert(12, f'space_headway', 0)
    data.insert(13, f's_diff', 0)

    data.columns = ['Trajectory_ID', 'Time_Index',
                    'ID_LV', 'Type_LV', 'Pos_LV', 'Speed_LV', 'Acc_LV',
                    'ID_FAV', 'Pos_FAV', 'Speed_FAV', 'Acc_FAV',
                    'Spatial_Gap', 'Spatial_Headway', 'Speed_Diff']

    data['Speed_Diff'] = data['Speed_LV'] - data['Speed_FAV']
    data['Spatial_Headway'] = data['Spatial_Gap'] + default_vehicle_length
    

    data = data.reset_index(drop=True)

    data['Acc_LV'] = ((data['Speed_LV'] - data['Speed_LV'].shift(1)) / 0.1).shift(-1)
    data['Acc_FAV'] = ((data['Speed_FAV'] - data['Speed_FAV'].shift(1)) / 0.1).shift(-1)

    average_speed = (data['Speed_FAV'] + data['Speed_FAV'].shift(1)) / 2
    data['Pos_FAV'] = (0.1 * average_speed).cumsum()
    data.loc[0, 'Pos_FAV'] = 0
    data['Pos_LV'] = data['Pos_FAV'] + data['Spatial_Headway']
    data = data.iloc[:-1]

    data['Time_Index'] -= data['Time_Index'].min()

    data.to_csv(output_path, index=False)


def Ohio_single_convert_format(input_path, output_path):
    related_columns = ['ID', 'Time', 'pos_x_av_f', 'speed_av', 'acc_av', 'pos_x_sv_f', 'speed_sv', 'acc_sv',
                       'closest_distance_longitudinal (gap)', 'distance_av (headway)', 'lane_id_av', 'lane_id_sv']

    data = pd.read_csv(input_path, usecols=related_columns)

    data = data[data['lane_id_av'] == data['lane_id_sv']]
    data.drop(columns=['lane_id_sv', 'lane_id_av'], inplace=True)

    data = data[data['pos_x_av_f'] > data['pos_x_sv_f']]

    data['reset'] = data['ID'].diff() < 0
    data['traj_id'] = data['reset'].cumsum() + 1
    data.drop(columns=['reset'], inplace=True)

    idx = data.groupby(['traj_id', 'Time'])['closest_distance_longitudinal (gap)'].idxmin()
    follower_ids = data.loc[idx, ['traj_id', 'Time', 'ID']].set_index(['traj_id', 'Time'])
    data['tmp_index'] = data.index
    data = data.merge(follower_ids, on=['traj_id', 'Time'], how='left', suffixes=('', '_follower'))
    data.rename(columns={'ID_follower': 'follower_id'}, inplace=True)
    data.drop(columns=['tmp_index'], inplace=True)

    data = data[data['ID'] == data['follower_id']]
    data.drop(columns=['follower_id'], inplace=True)

    new_traj_id = 0
    traj_id_mapping = {}

    data.sort_values(by=['traj_id', 'ID'], inplace=True)
    for index, row in data.iterrows():
        key = (row['traj_id'], row['ID'])
        if key not in traj_id_mapping:
            new_traj_id += 1
            traj_id_mapping[key] = new_traj_id
        data.at[index, 'traj_id'] = traj_id_mapping[key]

    cols = ['traj_id'] + [col for col in data.columns if col != 'traj_id']
    data = data[cols]

    data['ID'] = -1
    data['traj_id'] -= 1

    data.insert(3, 'Type_LV', -1)
    data.insert(8, 'ID_FAV', 0)
    column_data = data['distance_av (headway)']
    data.drop(columns=['distance_av (headway)'], inplace=True)
    data.insert(12, 'distance_av (headway)', column_data)
    data.insert(13, f's_diff', 0)

    data.columns = ['Trajectory_ID', 'Time_Index',
                    'ID_LV', 'Type_LV', 'Pos_LV', 'Speed_LV', 'Acc_LV',
                    'ID_FAV', 'Pos_FAV', 'Speed_FAV', 'Acc_FAV',
                    'Spatial_Gap', 'Spatial_Headway', 'Speed_Diff']

    data['Time_Index'], data['ID_LV'] = data['ID_LV'], data['Time_Index']

    data['Speed_Diff'] = data['Speed_LV'] - data['Speed_FAV']

    data.to_csv(output_path, index=False)


def Ohio_two_convert_format(input_path, output_path):
    related_columns = ['ID', 'Time', 'pos_x_av_f', 'speed_av', 'acc_av', 'pos_x_sv1_f', 'speed_sv1', 'acc_sv1',
                       'pos_x_sv2_f', 'speed_sv2', 'acc_sv2', 'lane_id_av', 'lane_id_sv1', 'lane_id_sv2', 'dim_x_av',
                       'dim_x_sv1', 'dim_x_sv2']

    def find_one_vehicle(id, id2):
        data = pd.read_csv(input_path, usecols=related_columns)

        data['ID'] += 1

        data['reset'] = data['ID'].diff() < 0
        data['traj_id'] = data['reset'].cumsum() + 1
        data.drop(columns=['reset'], inplace=True)

        result = data.groupby(['traj_id', 'Time']).first().reset_index()

        sv_to_av_columns = {
            f'pos_x_sv{id2}_f': 'pos_x_av_f',
            f'speed_sv{id2}': 'speed_av',
            f'acc_sv{id2}': 'acc_av',
            f'lane_id_sv{id2}': 'lane_id_av',
            f'dim_x_sv{id2}': 'dim_x_av'
        }
        df_sv_as_av = result[
            ['traj_id', 'Time', 'pos_x_sv1_f', 'speed_sv1', 'acc_sv1', 'pos_x_sv2_f', 'speed_sv2', 'acc_sv2',
             'lane_id_sv1', 'lane_id_sv2', 'dim_x_sv1', 'dim_x_sv2']].copy()
        df_sv_as_av.rename(columns=sv_to_av_columns, inplace=True)
        df_sv_as_av['ID'] = id2 - 1

        data = pd.concat([data, df_sv_as_av], ignore_index=True)
        data = data.sort_values(by=['traj_id', 'ID', 'Time']).reset_index(drop=True)
        data = data.drop(columns=list(sv_to_av_columns.keys()))

        data = data[data['lane_id_av'] == data[f'lane_id_sv{id}']]
        data.drop(columns=[f'lane_id_sv{id}', 'lane_id_av'], inplace=True)

        data = data[data['pos_x_av_f'] > data[f'pos_x_sv{id}_f']]

        data['space_headway'] = data['pos_x_av_f'] - data[f'pos_x_sv{id}_f']

        data['reset'] = data['ID'].diff() < 0
        data['traj_id'] = data['reset'].cumsum() + 1
        data.drop(columns=['reset'], inplace=True)

        idx = data.groupby(['traj_id', 'Time'])['space_headway'].idxmin()
        follower_ids = data.loc[idx, ['traj_id', 'Time', 'ID']].set_index(['traj_id', 'Time'])
        data['tmp_index'] = data.index
        data = data.merge(follower_ids, on=['traj_id', 'Time'], how='left', suffixes=('', '_follower'))
        data.rename(columns={'ID_follower': 'follower_id'}, inplace=True)
        data.drop(columns=['tmp_index'], inplace=True)

        data = data[data['ID'] == data['follower_id']]
        data.drop(columns=['follower_id'], inplace=True)

        new_traj_id = 0
        traj_id_mapping = {}

        data.sort_values(by=['traj_id', 'ID'], inplace=True)
        for index, row in data.iterrows():
            key = (row['traj_id'], row['ID'])
            if key not in traj_id_mapping:
                new_traj_id += 1
                traj_id_mapping[key] = new_traj_id
            data.at[index, 'traj_id'] = traj_id_mapping[key]

        cols = ['traj_id'] + [col for col in data.columns if col != 'traj_id']
        data = data[cols]

        data['traj_id'] -= 1

        data['ID'] = data['ID'].apply(lambda x: x if x in [0, 1] else -1)

        data.insert(3, 'Type_LV', 0)
        data.insert(8, 'ID_FAV', id - 1)
        data.insert(13, 'space_gap', 0)
        data.insert(15, 's_diff', 0)

        new_order = ['traj_id', 'Time', 'ID', 'Type_LV', 'pos_x_av_f',
                     'speed_av', 'acc_av', 'ID_FAV', f'pos_x_sv{id}_f',
                     f'speed_sv{id}', f'acc_sv{id}', 'space_gap', 'space_headway',
                     's_diff', 'dim_x_av', f'dim_x_sv{id}']
        data = data[new_order]

        data.columns = ['Trajectory_ID', 'Time_Index',
                        'ID_LV', 'Type_LV', 'Pos_LV', 'Speed_LV', 'Acc_LV',
                        'ID_FAV', 'Pos_FAV', 'Speed_FAV', 'Acc_FAV',
                        'Spatial_Gap', 'Spatial_Headway', 'Speed_Diff', 'Len_LV', 'Len_FAV']

        data['Type_LV'] = data['ID_LV'].apply(lambda x: 1 if x in [0, 1] else 0)
        data['Speed_Diff'] = data['Speed_LV'] - data['Speed_FAV']
        data['Spatial_Gap'] = data['Spatial_Headway'] - data['Len_LV'] / 2 - data['Len_FAV'] / 2
        

        data = data.drop(columns=['Len_LV', 'Len_FAV'])

        data.to_csv(output_path + f'_{id}.csv', index=False)

    find_one_vehicle(1, 2)
    find_one_vehicle(2, 1)


def Waymo_perception_convert_format(input_path, output_path):
    related_columns = ['segment_id', 'local_veh_id', 'length', 'local_time', 'follower_id', 'leader_id',
                       'processed_position', 'processed_speed', 'processed_accer']

    data = pd.read_csv(input_path, usecols=related_columns)
    data = data[data['follower_id'] == 0]

    def merge_rows(group):
        merged_df_list = []

        for local_time in group['local_time'].unique():
            temp_group = group[group['local_time'] == local_time]
            temp_dict = {'segment_id': temp_group['segment_id'].iloc[0], 'local_time': local_time}

            for _, row in temp_group.iterrows():
                suffix = '_f' if row['local_veh_id'] == row['follower_id'] else '_l'
                for col in ['length', 'processed_position', 'processed_speed', 'processed_accer']:
                    temp_dict[f'{col}{suffix}'] = row[col]
                temp_dict['follower_id'] = row['follower_id']
                temp_dict['leader_id'] = row['leader_id']

            merged_df_list.append(pd.DataFrame([temp_dict]))

        merged_df = pd.concat(merged_df_list, ignore_index=True)

        return merged_df

    data = data.groupby('segment_id').apply(merge_rows).reset_index(drop=True)

    new_order = [
        'segment_id', 'local_time',
        'leader_id', 'processed_position_l', 'processed_speed_l', 'processed_accer_l',
        'follower_id', 'processed_position_f', 'processed_speed_f', 'processed_accer_f',
        'length_l', 'length_f'
    ]

    data = data.reindex(columns=new_order)

    data.insert(3, 'type', 0)
    data.insert(11, 'space_gap', 0)
    data.insert(12, 'space_headway', 0)
    data.insert(13, 's_diff', 0)

    data.columns = ['Trajectory_ID', 'Time_Index',
                    'ID_LV', 'Type_LV', 'Pos_LV', 'Speed_LV', 'Acc_LV',
                    'ID_FAV', 'Pos_FAV', 'Speed_FAV', 'Acc_FAV',
                    'Spatial_Gap', 'Spatial_Headway', 'Speed_Diff',
                    'length_l', 'length_f']

    data['ID_LV'] = -1
    data['Type_LV'] = 0
    data['Speed_Diff'] = data['Speed_LV'] - data['Speed_FAV']
    data['Spatial_Headway'] = data['Pos_LV'] - data['Pos_FAV']
    data['Spatial_Gap'] = data['Spatial_Headway'] - data['length_l'] / 2 - data['length_f'] / 2

    data = data.drop(columns=['length_l', 'length_f'])

    data.to_csv(output_path, index=False)


def Waymo_motion_convert_format(input_path, output_path):
    data = pd.read_csv(input_path)

    for idx in range(4):
        data.insert(1 + idx, f'col_{1 + idx}', 0)
    for idx in range(3):
        data.insert(6 + idx, f'col_{5 + idx}', 0)
    data.insert(10, f'Acc_FAV', 0)
    data.insert(11, f'Spatial_Gap', 0)
    data.insert(12, f'Spatial_Headway', 0)
    data.insert(13, 'Speed_Diff', 0)

    data.columns = ['Trajectory_ID', 'Time_Index', 'ID_LV', 'Type_LV', 'Pos_LV', 'Speed_LV', 'Acc_LV',
                    'ID_FAV', 'Pos_FAV', 'Speed_FAV', 'Acc_FAV',
                    'Spatial_Gap', 'Spatial_Headway', 'Speed_Diff',
                    'leader_x', 'leader_y', 'leader_length', 'follower_x', 'follower_y', 'follower_length']

    data['Time_Index'] = data.groupby('Trajectory_ID').cumcount() / 10
    data['ID_LV'] = -1
    data['ID_FAV'] = 0

    data['Spatial_Headway'] = (np.sqrt(
        (data['leader_x'] - data['follower_x']) ** 2 + (data['leader_y'] - data['follower_y']) ** 2))
    data['Spatial_Gap'] = data['Spatial_Headway'] - data['leader_length'] / 2 - data['follower_length'] / 2
    data['Speed_Diff'] = data['Speed_LV'] - data['Speed_FAV']

    data = data.drop(['leader_x', 'leader_y', 'follower_x', 'follower_y', 'leader_length', 'follower_length'], axis=1)

    data['Acc_LV'] = ((data['Speed_LV'] - data['Speed_LV'].shift(1)) / 0.1).shift(-1)
    data['Acc_FAV'] = ((data['Speed_FAV'] - data['Speed_FAV'].shift(1)) / 0.1).shift(-1)

    average_speed = (data['Speed_FAV'] + data['Speed_FAV'].shift(1)) / 2
    data['Pos_FAV'] = (0.1 * average_speed).cumsum()
    data.loc[0, 'Pos_FAV'] = 0
    data['Pos_LV'] = data['Pos_FAV'] + data['Spatial_Headway']
    data = data.iloc[:-1]

    data.to_csv(output_path, index=False)


def Argoverse_convert_format(input_path, output_path):
    data = pd.read_csv(input_path)

    for idx in range(4):
        data.insert(1 + idx, f'col_{1 + idx}', 0)
    for idx in range(3):
        data.insert(6 + idx, f'col_{5 + idx}', 0)
    data.insert(10, f'Acc_FAV', 0)
    data.insert(11, f'Spatial_Gap', 0)
    data.insert(12, f'Spatial_Headway', 0)
    data.insert(13, 'Speed_Diff', 0)

    data.columns = ['Trajectory_ID', 'Time_Index', 'ID_LV', 'Type_LV', 'Pos_LV', 'Speed_LV', 'Acc_LV',
                    'ID_FAV', 'Pos_FAV', 'Speed_FAV', 'Acc_FAV',
                    'Spatial_Gap', 'Spatial_Headway', 'Speed_Diff',
                    'leader_x', 'leader_y', 'follower_x', 'follower_y']

    data['Time_Index'] = data.groupby('Trajectory_ID').cumcount() / 10
    data['ID_LV'] = -1
    data['Type_LV'] = 0
    data['ID_FAV'] = 0

    data['Spatial_Headway'] = (np.sqrt(
        (data['leader_x'] - data['follower_x']) ** 2 + (data['leader_y'] - data['follower_y']) ** 2))
    data['Spatial_Gap'] = data['Spatial_Headway'] - default_vehicle_length
    data['Speed_Diff'] = data['Speed_LV'] - data['Speed_FAV']

    data = data.drop(['leader_x', 'leader_y', 'follower_x', 'follower_y'], axis=1)

    data['Acc_LV'] = ((data['Speed_LV'] - data['Speed_LV'].shift(1)) / 0.1).shift(-1)
    data['Acc_FAV'] = ((data['Speed_FAV'] - data['Speed_FAV'].shift(1)) / 0.1).shift(-1)

    average_speed = (data['Speed_FAV'] + data['Speed_FAV'].shift(1)) / 2
    data['Pos_FAV'] = (0.1 * average_speed).cumsum()
    data.loc[0, 'Pos_FAV'] = 0
    data['Pos_LV'] = data['Pos_FAV'] + data['Spatial_Headway']
    data = data.iloc[:-1]

    data.to_csv(output_path, index=False)
