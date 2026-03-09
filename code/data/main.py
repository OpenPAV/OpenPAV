from CF_extraction import *
from data_transformation import *
from data_cleaning import *
from model_calibration import *
from data_analysis import *
from pathlib import Path


def Vanderbilt_two_vehicle_ACC():
    # Step 1: Convert dataset to a uniform car-following data format and analyze statistics.
    original_data_path = './Dataset/Vanderbilt/data/Two-vehicle ACC driving, Tennessee 2019/Processed_CAN_Data_a.csv'
    uniform_format_path = './Dataset/Vanderbilt/output/step1_two_vehicle_ACC.csv'
    Vanderbilt_convert_format(original_data_path, uniform_format_path)
    step1_stat_result_path = './Dataset/Vanderbilt/output/step1_analysis_two_vehicle_ACC'
    analyze_statistics(uniform_format_path, step1_stat_result_path)

    # Step 2: Clean data and revise trajectory IDs for further analysis.
    clean_data = fill_and_clean(uniform_format_path, 10, [None, 10, None, 10, 10, None, None],
                                1e10, -1e10, 1e10, -1e10,
                                1e10, -1e10, 1e10, -1e10, 1e10, -1e10)
    step2_data_path = f'./Dataset/Vanderbilt/output/step2_two_vehicle_ACC.csv'
    revise_traj_id(clean_data, step2_data_path, 0.1, 70, 0, 0)
    step2_stat_result_path = './Dataset/Vanderbilt/output/step2_analysis_two_vehicle_ACC'
    analyze_statistics(step2_data_path, step2_stat_result_path)

    # Step 3: Further clean and refine data, and prepare for performance analysis.
    clean_data = fill_and_clean(step2_data_path, 10, None,
                                120, 1e-5, 1e10, 0.1,
                                1e10, 0.1, 5, -5, 5, -5)
    step3_data_path = f'./Dataset/Vanderbilt/output/step3_two_vehicle_ACC.csv'
    revise_traj_id(clean_data, step3_data_path, 0.1, 70, 0, 0)
    step3_stat_result_path = './Dataset/Vanderbilt/output/step3_analysis_two_vehicle_ACC'
    analyze_statistics(step3_data_path, step3_stat_result_path)

    # Analysis
    performance_result_path = './Dataset/Vanderbilt/output/performance_metrics_two_vehicle_ACC.csv'
    analyze_AV_performance(step3_data_path, performance_result_path)

    scatter_plot_path = './Dataset/Vanderbilt/output/scatter_two_vehicle_ACC'
    draw_scatter(step3_data_path, scatter_plot_path)

    calibration_result_path = './Dataset/Vanderbilt/output/calibration_two_vehicle_ACC.csv'
    linear_regression = CFModelRegress(step3_data_path, 0.1)
    linear_regression.main(calibration_result_path, "linear")


def MicroSimACC():
    # Step 1: Convert dataset to a uniform car-following data format and analyze statistics.
    merge_data_list = []
    for p1 in [35, 60]:
        for p2 in [0, 15, 25, 35, 45, 50, 55]:
            if p2 > p1 - 10: continue
            for p3 in ['L', 'M', 'S']:
                for p4 in range(1, 5):
                    original_data_path = \
                        f'./Dataset/MicroSimACC/data/2-Vehicle ACC Car Following Experiments (CCF, Same Desired Speed)/{p1}_{p2}_{p3}_{p4}.csv'
                    uniform_format_path = f'./Dataset/MicroSimACC/output/step1_same_speed_{p1}_{p2}_{p3}_{p4}.csv'
                    MicroSimACC_convert_format(original_data_path, uniform_format_path)
                    merge_data_list.append(uniform_format_path)

    merge_data_path = f'./Dataset/MicroSimACC/output/step1_merge.csv'
    merge_data(merge_data_list, merge_data_path)
    step1_stat_result_path = './Dataset/MicroSimACC/output/step1_analysis'
    analyze_statistics(merge_data_path, step1_stat_result_path)

    # Step 2: Clean data and revise trajectory IDs for further analysis.
    clean_data = fill_and_clean(merge_data_path, 10, [None, 10, None, 10, 10, None, None],
                                1e10, -1e10, 1e10, -1e10,
                                1e10, -1e10, 1e10, -1e10, 1e10, -1e10)
    step2_data_path = f'./Dataset/MicroSimACC/output/step2.csv'
    revise_traj_id(clean_data, step2_data_path, 0.2, 70, 0, 0)
    step2_stat_result_path = './Dataset/MicroSimACC/output/step2_analysis'
    analyze_statistics(step2_data_path, step2_stat_result_path)

    # Step 3: Further clean and refine data, and prepare for performance analysis.
    clean_data = fill_and_clean(step2_data_path, 10, None,
                                120, 1e-5, 1e10, 0.1,
                                1e10, 0.1, 5, -5, 5, -5)
    step3_data_path = f'./Dataset/MicroSimACC/output/step3.csv'
    revise_traj_id(clean_data, step3_data_path, 0.2, 70, 0, 0)
    step3_stat_result_path = './Dataset/MicroSimACC/output/step3_analysis'
    analyze_statistics(step3_data_path, step3_stat_result_path)

    # Analysis
    performance_result_path = './Dataset/MicroSimACC/output/performance_metrics.csv'
    analyze_AV_performance(step3_data_path, performance_result_path)

    scatter_plot_path = './Dataset/MicroSimACC/output/scatter'
    draw_scatter(step3_data_path, scatter_plot_path)

    calibration_result_path = './Dataset/MicroSimACC/output/calibration.csv'
    linear_regression = CFModelRegress(step3_data_path, 0.1)
    linear_regression.main(calibration_result_path, "linear")


def CATS_ACC():
    # Step 1: Convert dataset to a uniform car-following data format and analyze statistics.
    merge_data_list = []
    for i in range(1, 6):
        original_data_path = f'./Dataset/CATS/data/ACC/test1118/test{i}.xlsx'
        uniform_format_path = f'./Dataset/CATS/output/step1_ACC_test1118_{i}.csv'
        CATSACC_convert_format(original_data_path, uniform_format_path)
        merge_data_list.append(uniform_format_path)
    for i in range(1, 9):
        original_data_path = f'./Dataset/CATS/data/ACC/test1124/test{i}.xlsx'
        uniform_format_path = f'./Dataset/CATS/output/step1_ACC_test1124_{i}.csv'
        CATSACC_convert_format(original_data_path, uniform_format_path)
        merge_data_list.append(uniform_format_path)

    merge_data_path = f'./Dataset/CATS/output/step1_ACC_merge.csv'
    merge_data(merge_data_list, merge_data_path)
    step1_stat_result_path = './Dataset/CATS/output/step1_analysis_ACC'
    analyze_statistics(merge_data_path, step1_stat_result_path)

    # Step 2: Clean data and revise trajectory IDs for further analysis.
    clean_data = fill_and_clean(merge_data_path, 10, [None, 10, None, 10, 10, None, None],
                                1e10, -1e10, 1e10, -1e10,
                                1e10, -1e10, 1e10, -1e10, 1e10, -1e10)
    step2_data_path = f'./Dataset/CATS/output/step2_ACC.csv'
    revise_traj_id(clean_data, step2_data_path, 0.1, 70, 0, 0)
    step2_stat_result_path = './Dataset/CATS/output/step2_analysis_ACC'
    analyze_statistics(step2_data_path, step2_stat_result_path)

    # Step 3: Further clean and refine data, and prepare for performance analysis.
    clean_data = fill_and_clean(step2_data_path, 10, None,
                                120, 1e-5, 1e10, 0.1,
                                1e10, 0.1, 5, -5, 5, -5)
    step3_data_path = f'./Dataset/CATS/output/step3_ACC.csv'
    revise_traj_id(clean_data, step3_data_path, 0.1, 70, 0, 0)
    step3_stat_result_path = './Dataset/CATS/output/step3_analysis_ACC'
    analyze_statistics(step3_data_path, step3_stat_result_path)

    # Analysis
    performance_result_path = './Dataset/CATS/output/performance_metrics_ACC.csv'
    analyze_AV_performance(step3_data_path, performance_result_path)
    scatter_plot_path = './Dataset/CATS/output/scatter_ACC'
    draw_scatter(step3_data_path, scatter_plot_path)

    calibration_result_path = './Dataset/CATS/output/calibration_ACC.csv'
    linear_regression = CFModelRegress(step3_data_path, 0.1)
    linear_regression.main(calibration_result_path, 'linear')


def CATS_platoon():
    # Step 1: Convert dataset to a uniform car-following data format and analyze statistics.
    original_data_path = f'./Dataset/CATS/data/Platoon'
    uniform_format_path = f'./Dataset/CATS/output/step1_platoon.csv'
    CATSPlatoon_convert_format(original_data_path, uniform_format_path)
    step1_stat_result_path = './Dataset/CATS/output/step1_analysis_platoon'
    analyze_statistics(uniform_format_path, step1_stat_result_path)

    # Step 2: Clean data and revise trajectory IDs for further analysis. 
    clean_data = fill_and_clean(uniform_format_path, 10, [None, 10, None, 10, 10, None, None],
                                1e10, -1e10, 1e10, -1e10,
                                1e10, -1e10, 1e10, -1e10, 1e10, -1e10)
    step2_data_path = f'./Dataset/CATS/output/step2_platoon.csv'
    revise_traj_id(clean_data, step2_data_path, 1, 90, 0, 0)
    step2_stat_result_path = './Dataset/CATS/output/step2_analysis_platoon'
    analyze_statistics(step2_data_path, step2_stat_result_path)

    # Step 3: Further clean and refine data, and prepare for performance analysis.
    clean_data = fill_and_clean(step2_data_path, 10, None,
                                120, 1e-5, 1e10, 0.1,
                                1e10, 0.1, 5, -5, 5, -5)
    step3_data_path = f'./Dataset/CATS/output/step3_platoon.csv'
    revise_traj_id(clean_data, step3_data_path, 1, 90, 0, 0)
    step3_stat_result_path = './Dataset/CATS/output/step3_analysis_platoon'
    analyze_statistics(step3_data_path, step3_stat_result_path)

    # Analysis
    performance_result_path = './Dataset/CATS/output/performance_metrics_platoon.csv'
    analyze_AV_performance(step3_data_path, performance_result_path)
    scatter_plot_path = './Dataset/CATS/output/scatter_platoon'
    draw_scatter(step3_data_path, scatter_plot_path)

    calibration_result_path = './Dataset/CATS/output/calibration_platoon.csv'
    linear_regression = CFModelRegress(step3_data_path, 1)
    linear_regression.main(calibration_result_path, 'linear')


def CATS_UWM():
    merge_data_list = []
    for i in range(1, 6):
        original_data_path = f'./Dataset/CATS/data/UWM/Test{i}.csv'
        uniform_format_path = f'./Dataset/CATS/output/step1_UWM_test{i}.csv'
        CATSUW_convert_format(original_data_path, uniform_format_path)
        merge_data_list.append(uniform_format_path)

    merge_data_path = f'./Dataset/CATS/output/step1_UWM_merge.csv'
    merge_data(merge_data_list, merge_data_path)
    step1_stat_result_path = './Dataset/CATS/output/step1_analysis_UWM'
    analyze_statistics(merge_data_path, step1_stat_result_path)

    clean_data = fill_and_clean(merge_data_path, 10, [None, 10, None, 10, 10, None, None],
                                1e10, -1e10, 1e10, -1e10,
                                1e10, -1e10, 1e10, -1e10, 1e10, -1e10)
    step2_data_path = f'./Dataset/CATS/output/step2_UWM.csv'
    revise_traj_id(clean_data, step2_data_path, 0.1, 70, 0, 0)
    step2_stat_result_path = './Dataset/CATS/output/step2_analysis_UWM'
    analyze_statistics(step2_data_path, step2_stat_result_path)

    # Step 3: Further clean and refine data, and prepare for performance analysis.
    clean_data = fill_and_clean(step2_data_path, 10, None,
                                120, 1e-5, 1e10, 0.1,
                                1e10, 0.1, 5, -5, 5, -5)
    step3_data_path = f'./Dataset/CATS/output/step3_UWM.csv'
    revise_traj_id(clean_data, step3_data_path, 0.1, 70, 0, 0)
    step3_stat_result_path = './Dataset/CATS/output/step3_analysis_UWM'
    analyze_statistics(step3_data_path, step3_stat_result_path)

    # Analysis
    performance_result_path = './Dataset/CATS/output/performance_metrics_UWM.csv'
    analyze_AV_performance(step3_data_path, performance_result_path)
    scatter_plot_path = './Dataset/CATS/output/scatter_UWM'
    draw_scatter(step3_data_path, scatter_plot_path)

    calibration_result_path = './Dataset/CATS/output/calibration_UWM.csv'
    linear_regression = CFModelRegress(step3_data_path, 0.1)
    linear_regression.main(calibration_result_path, 'linear')


def OpenACC_Casale():
    id_map = {
        "Hyundai": 0,
        "Rexton": 1
    }

    merge_data_list = []
    for i in range(3, 12):
        original_data_path = f'./Dataset/OpenACC/data/Casale/part{i}.csv'
        uniform_format_path = f'./Dataset/OpenACC/output/step1_Casale_{i}.csv'
        OpenACC_convert_format(original_data_path, uniform_format_path, id_map)
        merge_data_list.append(uniform_format_path)

    merge_data_path = f'./Dataset/OpenACC/output/step1_Casale_merge.csv'
    merge_data(merge_data_list, merge_data_path)
    step1_stat_result_path = './Dataset/OpenACC/output/step1_analysis_Casale'
    analyze_statistics(merge_data_path, step1_stat_result_path)

    clean_data = fill_and_clean(merge_data_path, 10, [None, 10, None, 10, 10, None, None],
                                1e10, -1e10, 1e10, -1e10,
                                1e10, -1e10, 1e10, -1e10, 1e10, -1e10)
    step2_data_path = './Dataset/OpenACC/output/step2_Casale.csv'
    revise_traj_id(clean_data, step2_data_path, 0.1, 70, 0, 0)
    step2_stat_result_path = './Dataset/OpenACC/output/step2_analysis_Casale'
    analyze_statistics(step2_data_path, step2_stat_result_path)

    # Step 3: Further clean and refine data, and prepare for performance analysis.
    clean_data = fill_and_clean(step2_data_path, 10, None,
                                120, 1e-5, 1e10, 0.1,
                                1e10, 0.1, 5, -5, 5, -5)
    step3_data_path = f'./Dataset/OpenACC/output/step3_Casale.csv'
    revise_traj_id(clean_data, step3_data_path, 0.1, 70, 0, 0)
    step3_stat_result_path = './Dataset/OpenACC/output/step3_analysis_Casale'
    analyze_statistics(step3_data_path, step3_stat_result_path)

    # Analysis
    performance_result_path = './Dataset/OpenACC/output/performance_metrics_Casale.csv'
    analyze_AV_performance(step3_data_path, performance_result_path)
    scatter_plot_path = './Dataset/OpenACC/output/scatter_Casale'
    draw_scatter(step3_data_path, scatter_plot_path)

    calibration_result_path = './Dataset/OpenACC/output/calibration_Casale.csv'
    linear_regression = CFModelRegress(step3_data_path, 0.1)
    linear_regression.main(calibration_result_path, 'linear')


def OpenACC_Vicolungo():
    file_names = ['JRC-VC_260219_part2', 'JRC-VC_260219_part3', 'JRC-VC_260219_part4_highway',
                  'VC-JRC_260219_part1', 'VC-JRC_260219_part2', 'VC-JRC_270219_part1', 'VC-JRC_270219_part2',
                  'VC-JRC_280219_part2', 'VC-JRC_280219_part3']
    id_map = {
        "Ford(S-Max)": 0,
        "KIA(Niro)": 1,
        "Mini(Cooper)": 2,
        "Mitsubishi(OutlanderPHEV)": 3,
        "Mitsubishi(SpaceStar)": 4,
        "Peugeot(3008GTLine)": 5,
        "VW(GolfE)": 6
    }

    merge_data_list = []
    for i in range(1, 8):
        original_data_path = f'./Dataset/OpenACC/data/Vicolungo/' + file_names[i] + '.csv'
        uniform_format_path = f'./Dataset/OpenACC/output/step1_Vicolungo_{i}.csv'
        OpenACC_convert_format(original_data_path, uniform_format_path, id_map)
        merge_data_list.append(uniform_format_path)

    merge_data_path = f'./Dataset/OpenACC/output/step1_Vicolungo_merge.csv'
    merge_data(merge_data_list, merge_data_path)
    step1_stat_result_path = './Dataset/OpenACC/output/step1_analysis_Vicolungo'
    analyze_statistics(merge_data_path, step1_stat_result_path, False)

    clean_data = fill_and_clean(merge_data_path, 10, [None, 10, None, 10, 10, None, None],
                                1e10, -1e10, 1e10, -1e10,
                                1e10, -1e10, 1e10, -1e10, 1e10, -1e10)
    step2_data_path = f'./Dataset/OpenACC/output/step2_Vicolungo.csv'
    revise_traj_id(clean_data, step2_data_path, 0.1, 70, 0, 0)
    step2_stat_result_path = F'./Dataset/OpenACC/output/step2_analysis_Vicolungo'
    analyze_statistics(step2_data_path, step2_stat_result_path)

    # Step 3: Further clean and refine data, and prepare for performance analysis.
    clean_data = fill_and_clean(step2_data_path, 10, None,
                                120, 1e-5, 1e10, 0.1,
                                1e10, 0.1, 5, -5, 5, -5)
    step3_data_path = f'./Dataset/OpenACC/output/step3_Vicolungo.csv'
    revise_traj_id(clean_data, step3_data_path, 0.1, 70, 0, 0)
    step3_stat_result_path = './Dataset/OpenACC/output/step3_analysis_Vicolungo'
    analyze_statistics(step3_data_path, step3_stat_result_path)

    # Analysis
    performance_result_path = './Dataset/OpenACC/output/performance_metrics_Vicolungo.csv'
    analyze_AV_performance(step3_data_path, performance_result_path)
    scatter_plot_path = './Dataset/OpenACC/output/scatter_Vicolungo'
    draw_scatter(step3_data_path, scatter_plot_path)

    calibration_result_path = './Dataset/OpenACC/output/calibration_Vicolungo.csv'
    linear_regression = CFModelRegress(step3_data_path, 0.1)
    linear_regression.main(calibration_result_path, 'linear')


def OpenACC_ASta():
    id_map = {
        "Audi(A6)": 0,
        "Audi(A8)": 1,
        "BMW(X5)": 2,
        "Mercedes(AClass)": 3,
        "Tesla(Model3)": 4
    }

    merge_data_list = []
    for i in range(1, 11):
        if i != 3 and i != 10:  # ignore human driving data
            original_data_path = f'./Dataset/OpenACC/data/ASta/ASta_platoon{i}.csv'
            uniform_format_path = f'./Dataset/OpenACC/output/step1_ASta_{i}.csv'
            OpenACC_convert_format(original_data_path, uniform_format_path, id_map)
            merge_data_list.append(uniform_format_path)

    merge_data_path = f'./Dataset/OpenACC/output/step1_ASta_merge.csv'
    merge_data(merge_data_list, merge_data_path)
    step1_stat_result_path = './Dataset/OpenACC/output/step1_analysis_ASta'
    analyze_statistics(merge_data_path, step1_stat_result_path)

    clean_data = fill_and_clean(merge_data_path, 10, [None, 10, None, 10, 10, None, None],
                                1e10, -1e10, 1e10, -1e10,
                                1e10, -1e10, 1e10, -1e10, 1e10, -1e10)
    step2_data_path = './Dataset/OpenACC/output/step2_ASta.csv'
    revise_traj_id(clean_data, step2_data_path, 0.1, 70, 0, 0)
    step2_stat_result_path = F'./Dataset/OpenACC/output/step2_analysis_ASta'
    analyze_statistics(step2_data_path, step2_stat_result_path)

    # Step 3: Further clean and refine data, and prepare for performance analysis.
    clean_data = fill_and_clean(step2_data_path, 10, None,
                                120, 1e-5, 1e10, 0.1,
                                1e10, 0.1, 5, -5, 5, -5)
    step3_data_path = f'./Dataset/OpenACC/output/step3_ASta.csv'
    revise_traj_id(clean_data, step3_data_path, 0.1, 70, 0, 0)
    step3_stat_result_path = './Dataset/OpenACC/output/step3_analysis_ASta'
    analyze_statistics(step3_data_path, step3_stat_result_path)

    # Analysis
    performance_result_path = './Dataset/OpenACC/output/performance_metrics_ASta.csv'
    analyze_AV_performance(step3_data_path, performance_result_path)
    scatter_plot_path = './Dataset/OpenACC/output/scatter_ASta'
    draw_scatter(step3_data_path, scatter_plot_path)

    calibration_result_path = './Dataset/OpenACC/output/calibration_ASta.csv'
    linear_regression = CFModelRegress(step3_data_path, 0.1)
    linear_regression.main(calibration_result_path, 'linear')


def OpenACC_ZalaZone():
    id_map = {
        "AUDI_A4": 0,
        "AUDI_E_TRON": 1,
        "BMW_I3": 2,
        "JAGUAR_I_PACE": 3,
        "MAZDA_3": 4,
        "MERCEDES_GLE450": 5,
        "SMART_TARGET": 6,
        "SKODA_TARGET": 7,
        "TESLA_MODEL3": 8,
        "TESLA_MODELS": 9,
        "TESLA_MODELX": 10,
        "TOYOTA_RAV4": 11
    }

    merge_data_list = []
    for i in range(1, 27):
        original_data_path = f'./Dataset/OpenACC/data/ZalaZone/dynamic_part{i}.csv'
        uniform_format_path = f'./Dataset/OpenACC/output/step1_ZalaZone_dynamic_{i}.csv'
        OpenACC_convert_format(original_data_path, uniform_format_path, id_map)
        merge_data_list.append(uniform_format_path)

    for i in range(1, 48):
        if i != 30 and i != 40 and i != 43:  # ignore human driving data
            original_data_path = f'./Dataset/OpenACC/data/ZalaZone/handling_part{i}.csv'
            uniform_format_path = f'./Dataset/OpenACC/output/step1_ZalaZone_handling_{i}.csv'
            OpenACC_convert_format(original_data_path, uniform_format_path, id_map)
            merge_data_list.append(uniform_format_path)

    merge_data_path = f'./Dataset/OpenACC/output/step1_ZalaZone_merge.csv'
    merge_data(merge_data_list, merge_data_path)
    step1_stat_result_path = './Dataset/OpenACC/output/step1_analysis_ZalaZone'
    analyze_statistics(merge_data_path, step1_stat_result_path)

    clean_data = fill_and_clean(merge_data_path, 10, [None, 10, None, 10, 10, None, None],
                                1e10, -1e10, 1e10, -1e10,
                                1e10, -1e10, 1e10, -1e10, 1e10, -1e10)
    step2_data_path = './Dataset/OpenACC/output/step2_ZalaZone.csv'
    revise_traj_id(clean_data, step2_data_path, 0.1, 70, 0, 0)
    step2_stat_result_path = F'./Dataset/OpenACC/output/step2_analysis_ZalaZone'
    analyze_statistics(step2_data_path, step2_stat_result_path)

    # Step 3: Further clean and refine data, and prepare for performance analysis.
    clean_data = fill_and_clean(step2_data_path, 10, None,
                                120, 1e-5, 1e10, 0.1,
                                1e10, 0.1, 5, -5, 5, -5)
    step3_data_path = f'./Dataset/OpenACC/output/step3_ZalaZone.csv'
    revise_traj_id(clean_data, step3_data_path, 0.1, 70, 0, 0)
    step3_stat_result_path = F'./Dataset/OpenACC/output/step3_analysis_ZalaZone'
    analyze_statistics(step3_data_path, step3_stat_result_path)

    # Analysis
    performance_result_path = './Dataset/OpenACC/output/performance_metrics_ZalaZone.csv'
    analyze_AV_performance(step3_data_path, performance_result_path)
    scatter_plot_path = './Dataset/OpenACC/output/scatter_ZalaZone'
    draw_scatter(step3_data_path, scatter_plot_path)

    calibration_result_path = './Dataset/OpenACC/output/calibration_ZalaZone.csv'
    linear_regression = CFModelRegress(step3_data_path, 0.1)
    linear_regression.main(calibration_result_path, 'linear')


def Ohio_single_vehicle():
    original_data_path = './Dataset/Ohio/data/Advanced_Driver_Assistance_System__ADAS_-Equipped_Single-Vehicle_Data_for_Central_Ohio.csv'
    uniform_format_path = './Dataset/Ohio/output/step1_single_vehicle.csv'
    Ohio_single_convert_format(original_data_path, uniform_format_path)
    step1_stat_result_path = './Dataset/Ohio/output/step1_analysis_single_vehicle'
    analyze_statistics(uniform_format_path, step1_stat_result_path)

    clean_data = fill_and_clean(uniform_format_path, 10, [None, 10, None, 10, 10, None, None],
                                1e10, -1e10, 1e10, -1e10,
                                1e10, -1e10, 1e10, -1e10, 1e10, -1e10)
    step2_data_path = f'./Dataset/Ohio/output/step2_single_vehicle.csv'
    revise_traj_id(clean_data, step2_data_path, 0.1, 70, 0, 0)
    step2_stat_result_path = f'./Dataset/Ohio/output/step2_analysis_single_vehicle'
    analyze_statistics(step2_data_path, step2_stat_result_path)

    # Step 3: Further clean and refine data, and prepare for performance analysis.
    clean_data = fill_and_clean(step2_data_path, 10, None,
                                120, 1e-5, 1e10, 0.1,
                                1e10, 0.1, 5, -5, 5, -5)
    step3_data_path = f'./Dataset/Ohio/output/step3_single_vehicle.csv'
    revise_traj_id(clean_data, step3_data_path, 0.1, 70, 0, 0)
    step3_stat_result_path = f'./Dataset/Ohio/output/step3_analysis_single_vehicle'
    analyze_statistics(step3_data_path, step3_stat_result_path)

    # Analysis
    performance_result_path = './Dataset/Ohio/output/performance_metrics_single_vehicle.csv'
    analyze_AV_performance(step3_data_path, performance_result_path)
    scatter_plot_path = './Dataset/Ohio/output/scatter_single_vehicle'
    draw_scatter(step3_data_path, scatter_plot_path)

    calibration_result_path = './Dataset/Ohio/output/calibration_single_vehicle.csv'
    linear_regression = CFModelRegress(step3_data_path, 0.1)
    linear_regression.main(calibration_result_path, 'linear')


def Ohio_two_vehicle():
    original_data_path = './Dataset/Ohio/data/Advanced_Driver_Assistance_System__ADAS_-Equipped_Two-Vehicle_Data_for_Central_Ohio.csv'
    uniform_format_path = './Dataset/Ohio/output/step1_two_vehicle'
    Ohio_two_convert_format(original_data_path, uniform_format_path)
    merge_data_list = []
    for i in range(1, 3):
        merge_data_list.append(uniform_format_path + f'_{i}.csv')

    merge_data_path = './Dataset/Ohio/output/step1_two_vehicle_merge.csv'
    merge_data(merge_data_list, merge_data_path)
    step1_stat_result_path = './Dataset/Ohio/output/step1_analysis_two_vehicle'
    analyze_statistics(merge_data_path, step1_stat_result_path)

    clean_data = fill_and_clean(merge_data_path, 10, [None, 10, None, 10, 10, None, None],
                                1e10, -1e10, 1e10, -1e10,
                                1e10, -1e10, 1e10, -1e10, 1e10, -1e10)
    step2_data_path = './Dataset/Ohio/output/step2_two_vehicle.csv'
    revise_traj_id(clean_data, step2_data_path, 0.1, 70, 0, 0)
    step2_stat_result_path = './Dataset/Ohio/output/step2_analysis_two_vehicle'
    analyze_statistics(step2_data_path, step2_stat_result_path)

    # Step 3: Further clean and refine data, and prepare for performance analysis.
    clean_data = fill_and_clean(step2_data_path, 10, None,
                                120, 1e-5, 1e10, 0.1,
                                1e10, 0.1, 5, -5, 5, -5)
    step3_data_path = f'./Dataset/Ohio/output/step3_two_vehicle.csv'
    revise_traj_id(clean_data, step3_data_path, 0.1, 70, 0, 0)
    step3_stat_result_path = './Dataset/Ohio/output/step3_analysis_two_vehicle'
    analyze_statistics(step3_data_path, step3_stat_result_path)

    # Analysis
    performance_result_path = './Dataset/Ohio/output/performance_metrics_two_vehicle.csv'
    analyze_AV_performance(step3_data_path, performance_result_path)
    scatter_plot_path = './Dataset/Ohio/output/scatter_two_vehicle'
    draw_scatter(step3_data_path, scatter_plot_path)

    calibration_result_path = './Dataset/Ohio/output/calibration_two_vehicle.csv'
    linear_regression = CFModelRegress(step3_data_path, 0.1)
    linear_regression.main(calibration_result_path, 'linear')


def Waymo_perception():
    original_data_path = './Dataset/Waymo/data/Perception/car_following_trajectory.csv'
    uniform_format_path = './Dataset/Waymo/output/step1_perception.csv'
    Waymo_perception_convert_format(original_data_path, uniform_format_path)
    step1_stat_result_path = './Dataset/Waymo/output/step1_analysis_perception'
    analyze_statistics(uniform_format_path, step1_stat_result_path)

    clean_data = fill_and_clean(uniform_format_path, 10, [None, 10, None, 10, 10, None, None],
                                1e10, -1e10, 1e10, -1e10,
                                1e10, -1e10, 1e10, -1e10, 1e10, -1e10)
    step2_data_path = './Dataset/Waymo/output/step2_perception.csv'
    revise_traj_id(clean_data, step2_data_path, 0.1, 70, 0, 0)
    step2_stat_result_path = './Dataset/Waymo/output/step2_analysis_perception'
    analyze_statistics(step2_data_path, step2_stat_result_path)

    # Step 3: Further clean and refine data, and prepare for performance analysis.
    clean_data = fill_and_clean(step2_data_path, 10, None,
                                120, 1e-5, 1e10, 0.1,
                                1e10, 0.1, 5, -5, 5, -5)
    step3_data_path = f'./Dataset/Waymo/output/step3_perception.csv'
    revise_traj_id(clean_data, step3_data_path, 0.1, 70, 0, 0)
    step3_stat_result_path = './Dataset/Waymo/output/step3_analysis_perception'
    analyze_statistics(step3_data_path, step3_stat_result_path)

    # Analysis
    performance_result_path = './Dataset/Waymo/output/performance_metrics_perception.csv'
    analyze_AV_performance(step3_data_path, performance_result_path)
    scatter_plot_path = './Dataset/Waymo/output/scatter_perception'
    draw_scatter(step3_data_path, scatter_plot_path)

    calibration_result_path = './Dataset/Waymo/output/calibration_perception.csv'
    linear_regression = CFModelRegress(step3_data_path, 0.1)
    linear_regression.main(calibration_result_path, 'linear')


def Waymo_motion():
    # Step 1: Convert Vanderbilt dataset to a uniform car-following data format and analyze statistics.
    cf_trajectory_list = []
    # for i in range(1000):
    #     original_data_path = (
    #             f'./Dataset/Waymo/data/Motion/uncompressed_tf_example_training_training_tfexample.tfrecord-'
    #             + '{:05}'.format(i) + '-of-01000')
    #     print("recording " + '{:05}'.format(i) + " ......")
    #     original_data = Waymo_extract_df(original_data_path)
    #     cf_trajectory_list.append(Waymo_extract_cf_traj(original_data))
    #     if (i + 1) % 100 == 0:
    #         cf_path = f'./Dataset/Waymo/output/step0_CF_trajectory_motion_{i - 99}-{i}.csv'
    #         combined_df = pd.concat(cf_trajectory_list, axis=0, ignore_index=True)
    #         combined_df = combined_df.reset_index()
    #         combined_df['traj_id'] = combined_df.index // 91
    #         combined_df.to_csv(cf_path, index=False)
    #         cf_trajectory_list = []

    merge_data_list = []
    for i in range(0, 1000, 100):
        cf_path = f'./Dataset/Waymo/output/step0_CF_trajectory_motion_{i}-{i + 99}.csv'
        uniform_format_path = f'./Dataset/Waymo/output/step1_motion_{i / 100}.csv'
        Waymo_motion_convert_format(cf_path, uniform_format_path)
        merge_data_list.append(uniform_format_path)

    merge_data_path = f'./Dataset/Waymo/output/step1_motion_merge.csv'
    merge_data(merge_data_list, merge_data_path)
    step1_stat_result_path = './Dataset/Waymo/output/step1_analysis_motion'
    analyze_statistics(merge_data_path, step1_stat_result_path)

    # Step 2: Clean data and revise trajectory IDs for further analysis.
    clean_data = fill_and_clean(merge_data_path, 10, [None, 10, None, 10, 10, None, None],
                                1e10, -1e10, 1e10, -1e10,
                                1e10, -1e10, 1e10, -1e10, 1e10, -1e10)
    step2_data_path = f'./Dataset/Waymo/output/step2_motion.csv'
    revise_traj_id(clean_data, step2_data_path, 0.1, 70, 0, 0, False)
    step2_stat_result_path = f'./Dataset/Waymo/output/step2_analysis_motion'
    analyze_statistics(step2_data_path, step2_stat_result_path)

    # Step 3: Further clean and refine data, and prepare for performance analysis.
    clean_data = fill_and_clean(step2_data_path, 10, None,
                                120, 1e-5, 1e10, 0.1,
                                1e10, 0.1, 5, -5, 5, -5)
    step3_data_path = './Dataset/Waymo/output/step3_motion.csv'
    revise_traj_id(clean_data, step3_data_path, 0.1, 70, 0, 0)
    step3_stat_result_path = f'./Dataset/Waymo/output/step3_analysis_motion'
    analyze_statistics(step3_data_path, step3_stat_result_path)

    # Analysis
    performance_result_path = './Dataset/Waymo/output/performance_metrics_motion.csv'
    analyze_AV_performance(step3_data_path, performance_result_path)
    scatter_plot_path = './Dataset/Waymo/output/scatter_motion'
    draw_scatter(step3_data_path, scatter_plot_path)

    calibration_result_path = './Dataset/Waymo/output/calibration_motion.csv'
    linear_regression = CFModelRegress(step3_data_path, 0.1)
    linear_regression.main(calibration_result_path, 'linear')


def Argoverse2():
    # Step 1: Convert Vanderbilt dataset to a uniform car-following data format and analyze statistics.
    directory_path = Path('./Dataset/Argoverse/data/val')
    cf_trajectory_list = []
    count = 1
    for original_data_path in directory_path.rglob('*'):
        original_data_path = str(original_data_path) + '/scenario_' + str(original_data_path)[-36:] + '.parquet'
        cf_trajectory_list.append(original_data_path)
        if len(cf_trajectory_list) == 25000:
            print(f"recording {count * len(cf_trajectory_list)} files......")
            original_data = Argo2_extract_df(cf_trajectory_list, 0)
            cf_path = f'./Dataset/Argoverse/output/CF_trajectories_{count}.csv'
            Argo2_extract_cf_traj(original_data, cf_path)
            cf_trajectory_list = []
            count += 1

    merge_data_list = []
    for i in range(1, 10):
        cf_path = f'./Dataset/Argoverse/output/step0_CF_trajectory_{i}.csv'
        uniform_format_path = f'./Dataset/Argoverse/output/step1_{i}.csv'
        Argoverse_convert_format(cf_path, uniform_format_path)
        merge_data_list.append(uniform_format_path)

    merge_data_path = f'./Dataset/Argoverse/output/step1_merge.csv'
    merge_data(merge_data_list, merge_data_path)
    step1_stat_result_path = './Dataset/Argoverse/output/step1_analysis'
    analyze_statistics(merge_data_path, step1_stat_result_path)

    # Step 2: Clean data and revise trajectory IDs for further analysis.
    clean_data = fill_and_clean(merge_data_path, 10, [None, 10, None, 10, 10, None],
                                1e10, -1e10, 1e10, -1e10,
                                1e10, -1e10, 1e10, -1e10, 1e10, -1e10)
    step2_data_path = f'./Dataset/Argoverse/output/step2.csv'
    revise_traj_id(clean_data, step2_data_path, 0.1, 70, 0, 0)
    step2_stat_result_path = './Dataset/Argoverse/output/step2_analysis'
    analyze_statistics(step2_data_path, step2_stat_result_path)

    # Step 3: Further clean and refine data, and prepare for performance analysis.
    clean_data = fill_and_clean(step2_data_path, 10, None,
                                120, 1e-5, 1e10, 0.1,
                                1e10, 0.1, 5, -5, 5, -5)
    step3_data_path = f'./Dataset/Argoverse/output/step3.csv'
    revise_traj_id(clean_data, step3_data_path, 0.1, 70, 0, 0)
    step3_stat_result_path = './Dataset/Argoverse/output/step3_analysis'
    analyze_statistics(step3_data_path, step3_stat_result_path)

    # Analysis
    performance_result_path = './Dataset/Argoverse/output/performance_metrics.csv'
    analyze_AV_performance(step3_data_path, performance_result_path)
    scatter_plot_path = './Dataset/Argoverse/output/scatte'
    draw_scatter(step3_data_path, scatter_plot_path)

    calibration_result_path = './Dataset/Argoverse/output/calibration.csv'
    linear_regression = CFModelRegress(step3_data_path, 0.1)
    linear_regression.main(calibration_result_path, 'linear')


if __name__ == "__main__":
    # Main entry point for the data extraction and analysis.
    # Vanderbilt_two_vehicle_ACC()
    #
    # MicroSimACC()
    #
    # CATS_ACC()
    # CATS_platoon()
    # CATS_UWM()
    #
    # OpenACC_Casale()
    # OpenACC_Vicolungo()
    # OpenACC_ASta()
    # OpenACC_ZalaZone()
    #
    # Ohio_single_vehicle()
    # Ohio_two_vehicle()
    #
    # Argoverse2()
    #
    # Waymo_perception()
    Waymo_motion()

    # # Draw performance metrics distribution for various datasets.
    # paths = [
    #     './Dataset/Vanderbilt/output/performance_metrics_two_vehicle_ACC.csv',
    #     './Dataset/MicroSimACC/output/performance_metrics.csv',
    #     './Dataset/CATS/output/performance_metrics_ACC.csv',
    #     './Dataset/CATS/output/performance_metrics_platoon.csv',
    #     './Dataset/CATS/output/performance_metrics_UWM.csv',
    #     './Dataset/OpenACC/output/performance_metrics_Casale.csv',
    #     './Dataset/OpenACC/output/performance_metrics_Vicolungo.csv',
    #     './Dataset/OpenACC/output/performance_metrics_ASta.csv',
    #     './Dataset/OpenACC/output/performance_metrics_ZalaZone.csv',
    #     './Dataset/Ohio/output/performance_metrics_single_vehicle.csv',
    #     './Dataset/Ohio/output/performance_metrics_two_vehicle.csv',
    #     './Dataset/Waymo/output/performance_metrics_perception.csv',
    #     './Dataset/Waymo/output/performance_metrics_motion.csv',
    #     './Dataset/Argoverse/output/performance_metrics.csv'
    # ]
    # output_path = './Dataset/performance_metrics_'
    # dataset_labels = ["Vanderbilt ACC", "MicroSimACC", "CATS ACC", "CATS Platoon", "CATS UWM",
    #                   "OpenACC Casale", "OpenACC Vicolungo", "OpenACC ASta", "OpenACC ZalaZone",
    #                   "Ohio Single", "Ohio Two", "Waymo Perception", "Waymo Motion", "Argoverse2"]
    # draw_2D_perfromance_metrics(paths[0:5], output_path + '1_', dataset_labels[0:5])
    # draw_2D_perfromance_metrics(paths[5:9], output_path + '2_', dataset_labels[5:9])
    # draw_2D_perfromance_metrics(paths[9:], output_path + '3_', dataset_labels[9:])
    #
    # # Compile statistics summary and draw distribution.
    # output_path = './Analysis/statistics_summary_'
    # merge_statistics_results(output_path)
    # output_path = './Analysis/step3_distribution.png'
    # draw_statistics_distribution(output_path)
    #
    # # Analyze label statistics and correlation
    # paths = [
    #     './Dataset/Vanderbilt/output/step3_two_vehicle_ACC.csv',
    #     './Dataset/MicroSimACC/output/step3.csv',
    #     './Dataset/CATS/output/step3_ACC.csv',
    #     './Dataset/CATS/output/step3_platoon.csv',
    #     './Dataset/CATS/output/step3_UWM.csv',
    #     './Dataset/OpenACC/output/step3_Casale.csv',
    #     './Dataset/OpenACC/output/step3_Vicolungo.csv',
    #     './Dataset/OpenACC/output/step3_ASta.csv',
    #     './Dataset/OpenACC/output/step3_ZalaZone.csv',
    #     './Dataset/Ohio/output/step3_single_vehicle.csv',
    #     './Dataset/Ohio/output/step3_two_vehicle.csv',
    #     './Dataset/Waymo/output/step3_perception.csv',
    #     './Dataset/Waymo/output/step3_motion.csv',
    #     './Dataset/Argoverse/output/step3.csv'
    # ]
    # output_path = f'./Analysis/labels_statistics_'
    # dataset_labels = ["Vanderbilt ACC", "MicroSimACC", "CATS ACC", "CATS Platoon", "CATS UWM",
    #                   "OpenACC Casale", "OpenACC Vicolungo", "OpenACC ASta", "OpenACC ZalaZone",
    #                   "Ohio Single", "Ohio Two", "Waymo Perception", "Waymo Motion", "Argoverse2"]
    # draw_2D_labels_statistics(paths[0:5], output_path + '1_', dataset_labels[0:5])
    # draw_2D_labels_statistics(paths[5:9], output_path + '2_', dataset_labels[5:9])
    # draw_2D_labels_statistics(paths[9:], output_path + '3_', dataset_labels[9:])
    #
    # # Analyze label statistics and calculate correlation.
    # output_path = './Analysis/correlation.csv'
    # correlation(output_path)
