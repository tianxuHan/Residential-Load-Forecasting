import pandas as pd
import numpy as np


def threshold2(th, adj):
    adj[adj < 0] = 0
    n = len(adj)
    I = np.eye(n)
    adj = adj - I
    a = np.max(adj)
    adj[adj == 0] = 10
    b = np.min(adj)
    adj[adj == 10] = 0
    adj = (adj - b) / (b - a)
    adj[adj < th] = 0
    adj = adj + I
    return adj


def threshold1(th, adj):
    adj[adj < 0] = 0
    n = len(adj)
    I = np.eye(n)
    a = np.max(adj)
    adj[adj == 0] = 10
    b = np.min(adj)
    adj[adj == 10] = 0
    adj = (adj - b) / (b - a)
    adj[adj < th] = 0
    adj = adj + I
    return adj

# adj2=generate_prs(rd,adj_save,prs_save)
# adj1=generate_te(rd,adj_save,te_save)

def load_data(dataset):
    sz_adj = pd.read_csv(r'D:\fuxian\data\TE114.csv', header=None)
    adj1 = np.mat(sz_adj)
    sz_adj2 = pd.read_csv(r'D:\fuxian\data\PCC114.csv', header=None)
    adj2 = np.mat(sz_adj2)
    data = pd.read_csv(r'D:\fuxian\data\data.csv',header=None)
    return data, adj1, adj2


def preprocess_data(data, time_len, trate, vrate, seq_len, pre_len):
    train_size = int(time_len * trate)
    vali_size = int(time_len * vrate)
    train_data = data[0:train_size]
    vali_date = data[train_size:train_size + vali_size]
    test_data = data[train_size + vali_size:time_len]
    trainX, trainY, valiX, valiY, testX, testY = [], [], [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len])
    for i in range(len(vali_date) - seq_len - pre_len):
        a = vali_date[i: i + seq_len + pre_len]
        valiX.append(a[0: seq_len])
        valiY.append(a[seq_len: seq_len + pre_len])
    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len])
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    valiX1 = np.array(valiX)
    valiY1 = np.array(valiY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, valiX1, valiY1, testX1, testY1
