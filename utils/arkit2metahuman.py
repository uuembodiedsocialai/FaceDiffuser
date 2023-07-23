import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

def convert_to_metahuman_format(anim_data, save_file):
    # converts npy animations outputted by the model into a format that can be used
    # to animate metahumans (or other arkit based characters)  in Unreal Engine
    col_names = [
        "Timecode", "BlendShapeCount", "BrowDownLeft", "BrowDownRight", "BrowInnerUp",
        "BrowOuterUpLeft", "BrowOuterUpRight", "CheekPuff", "CheekSquintLeft",
        "CheekSquintRight", "EyeBlinkLeft", "EyeBlinkRight", "EyeLookDownLeft",
        "EyeLookDownRight", "EyeLookInLeft", "EyeLookInRight", "EyeLookOutLeft",
        "EyeLookOutRight", "EyeLookUpLeft", "EyeLookUpRight", "EyeSquintLeft",
        "EyeSquintRight", "EyeWideLeft", "EyeWideRight", "JawForward", "JawLeft",
        "JawOpen", "JawRight", "MouthClose", "MouthDimpleLeft", "MouthDimpleRight",
        "MouthFrownLeft", "MouthFrownRight", "MouthFunnel", "MouthLeft",
        "MouthLowerDownLeft", "MouthLowerDownRight", "MouthPressLeft", "MouthPressRight",
        "MouthPucker", "MouthRight", "MouthRollLower", "MouthRollUpper", "MouthShrugLower",
        "MouthShrugUpper", "MouthSmileLeft", "MouthSmileRight", "MouthStretchLeft",
        "MouthStretchRight", "MouthUpperUpLeft", "MouthUpperUpRight", "NoseSneerLeft",
        "NoseSneerRight"
    ]

    additional_rows = []
    date = datetime.min
    for _ in range(anim_data.shape[0]):
        time_str =  date.strftime('%H:%M:%S') + f":{int(date.strftime('%f')) / 1000000 * 60:06.3f}"
        bs_num = 51
        new_row = [time_str, bs_num]
        date = date + timedelta(milliseconds=1000/30)
        additional_rows.append(new_row)

    new_anim_data = np.concatenate((np.array(additional_rows), anim_data), axis=1)
    df = pd.DataFrame(new_anim_data, columns=col_names)

    # adjust mouth closure
    # df["JawOpen"] = df["JawOpen"].astype(float) - 0.1

    df.to_csv(save_file, index=False)


if __name__ == '__main__':
    s_dir = 'results/beat' # source directory where the npy files are
    t_dir = 'results/beat_mh' # location where the results will be saved as csv

    if not os.path.exists(t_dir):
        os.makedirs(t_dir)
    for file in os.listdir(s_dir):
        data = np.load(os.path.join(s_dir, file))
        convert_to_metahuman_format(data, os.path.join(t_dir, f'{file.split(".")[0]}.csv'))
