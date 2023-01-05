import pandas as pd


def correct_csv(path: str, save_path: str):
    """
    function to refactor csv from UR to standard csv

    Params:
        path: path to the measurement csv
        save_path: path to where the standard csv should be saved
    """
    raw_data = pd.read_csv(path, sep=" ")
    raw_data.to_csv(save_path, index=False)


def filter_df(df: pd.DataFrame, drop_list: list):
    """
    function to get rid of useless columns in csv

    Params:
        df: dataframe to be cleaned
        drop_list: list of column names that should be dropped

    Returns:
        cleaned_df: dataframe without the columns in drop_list
    """
    return df.drop(drop_list, axis=1)


def flatten_df(df: pd.DataFrame):
    """
    function that checks if columns are supposed to be horizontal lines, and assigns them the value 0

    Params:
        df: dataframe that should be checked and processed

    Returns:
        flat_df: dataframe that maybe has flattened columns
    """
    for j in df.columns:
        if check_if_flatline(df[j]):
            df.loc[:len(df[j]), [j]] = [0]
    flat_df = df
    return flat_df


def downsampling(ptp: int, df):
    """
    function to downsample a dataframe to create useable patterns in current/voltage.
    reduces the frequency by only taking the ptp´t sample in a dataframe

    Params:
        ptp: interval in which dataframe should be resampled
        df: dataframe to resample

    Returns:
        dws_df: downsampled
    """
    dws_df = pd.DataFrame()
    for i in df.columns:
        dws_df[i] = df[i][::ptp]

    return dws_df


def check_if_flatline(df):
    """
    function that calculates the difference between two points in a dataframe column, to check if it resembles a horizontal line.
    if difference is smaller than 0.001, then column is a horizontal line.

    Params:
        df: column of a dataframe that should be checked if it resembles a horizontal line

    Returns:
        bool: True when column resembles a horizontal line, False if not
    """
    comp_len = int(len(df) / 10)
    lin_list = []
    for i in range(0, comp_len):
        near_dis = abs(df[i+5])-abs(df[i])
        if near_dis > 0.0001:
            lin_list.append("not Flat")

    if len(lin_list) == 0:
        return True
    else:
        return False


if __name__ == "__main__":
    path = r"I:\BA\Code\datasets\ur10_MTL_var_1_raw.csv"
    save_path = r"I:\BA\Code\datasets\ur10_MTL_var_1.csv"
    data = pd.read_csv(save_path)
    drop_list = ["Unnamed: 0", "target_q_0","target_q_1", "target_q_2", "target_q_3", "target_q_4", "target_q_5",
                 "target_qd_0", "target_qd_1", "target_qd_2", "target_qd_3", "target_qd_4", "target_qd_5", "target_current_0",
                 "target_current_1", "target_current_2", "target_current_3", "target_current_4", "target_current_5", "target_moment_0",
                 "target_moment_1", "target_moment_2", "target_moment_3", "target_moment_4", "target_moment_5", "timestamp",
                 "target_TCP_pose_0", "target_TCP_pose_1", "target_TCP_pose_2", "target_TCP_pose_3", "target_TCP_pose_4", "target_TCP_pose_5",
                 "target_TCP_speed_0", "target_TCP_speed_1", "target_TCP_speed_2", "target_TCP_speed_3", "target_TCP_speed_4", "target_TCP_speed_5",
                 "actual_digital_input_bits", "actual_execution_time", "robot_mode", "joint_mode_0", "joint_mode_1", "joint_mode_1", "joint_mode_2",
                 "joint_mode_3", "joint_mode_4", "joint_mode_5", "safety_mode", "actual_digital_output_bits", "runtime_state",
                 "output_int_register_0", "output_int_register_1", "output_int_register_2", "output_int_register_3", "output_int_register_4",
                 "output_int_register_5", "output_int_register_6", "output_int_register_7", "output_int_register_8", "output_int_register_9",
                 "output_int_register_10", "output_int_register_11", "output_int_register_12", "output_int_register_13", "output_int_register_14",
                "output_int_register_15","output_int_register_16", "output_int_register_17", "output_int_register_18", "output_int_register_19",
                 "output_int_register_20", "output_int_register_21", "output_int_register_22", "output_int_register_23",
                 "output_double_register_0", "output_double_register_1", "output_double_register_2", "output_double_register_3",
                 "output_double_register_4",
                 "output_double_register_5", "output_double_register_6", "output_double_register_7", "output_double_register_8",
                 "output_double_register_9",
                 "output_double_register_10", "output_double_register_11", "output_double_register_12", "output_double_register_13",
                 "output_double_register_14",
                 "output_double_register_15", "output_double_register_16", "output_double_register_17", "output_double_register_18",
                 "output_double_register_19",
                 "output_double_register_20", "output_double_register_21", "output_double_register_22", "output_double_register_23",
                 "target_speed_fraction", "speed_scaling"
                 ]
    drop_list1 = ["Unnamed: 0", "target_q_0","target_q_1", "target_q_2", "target_q_3", "target_q_4", "target_q_5",
                 "target_qd_0", "target_qd_1", "target_qd_2", "target_qd_3", "target_qd_4", "target_qd_5", "target_current_0",
                 "target_current_1", "target_current_2", "target_current_3", "target_current_4", "target_current_5", "target_moment_0",
                 "target_moment_1", "target_moment_2", "target_moment_3", "target_moment_4", "target_moment_5", "timestamp",
                 "target_TCP_pose_0", "target_TCP_pose_1", "target_TCP_pose_2", "target_TCP_pose_3", "target_TCP_pose_4", "target_TCP_pose_5",
                 "target_TCP_speed_0", "target_TCP_speed_1", "target_TCP_speed_2", "target_TCP_speed_3", "target_TCP_speed_4", "target_TCP_speed_5",
                 "actual_digital_input_bits", "actual_execution_time", "robot_mode", "joint_mode_0", "joint_mode_1", "joint_mode_1", "joint_mode_2",
                 "joint_mode_3", "joint_mode_4", "joint_mode_5", "safety_mode", "actual_digital_output_bits", "runtime_state"]


    #data = data.drop(["Unnamed: 0", "joint_temperatures_0", "joint_temperatures_1", "joint_temperatures_2", "joint_temperatures_3",
     #                                 "joint_temperatures_4","joint_temperatures_5"], axis=1)
    data = data.iloc[:65000, :]
    data.to_csv(save_path, index=False)
