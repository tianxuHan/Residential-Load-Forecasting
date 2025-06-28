import os
import pandas as pd
import numpy as np
import random


def all_csv(file_PATH, save_file):
    df_list = []
    tqdm = os.listdir(file_PATH)
    for i in range(0, len(tqdm)):
        files_path = os.path.join(file_PATH, tqdm[i])
        df = pd.read_csv(files_path)

        if i == 0:
            data = df.iloc[:, 1:2]
        else:
            data = df.iloc[:, 1]
        df_list.append(data)
    df2 = pd.concat(df_list, axis=1)
    save_path = os.path.join(file_PATH, save_file)
    df2.to_csv(save_path, index=False)


def rd_con(rd, read_path, save_path):
    df_list = []
    df = pd.read_csv(read_path, header=None)
    for i in range(0, len(rd)):
        data = df.iloc[:, rd[i]]
        df_list.append(data)
    df2 = pd.concat(df_list, axis=1)
    df2.to_csv(save_path, header=False, index=False)


def all_data(all_source, all_save, all7d):
    all_csv(file_PATH=all_source, save_file=all_save)
    data = pd.read_csv(os.path.join(all_source, all_save))
    data1 = np.asmatrix(data, dtype=np.float32)
    data1 = data1[262020:272100, 0:114]
    data1 = np.array(data1)
    data1 = pd.DataFrame(data1)
    data1.to_csv(all7d, index=False, header=False)
    print("all_data complete!")








