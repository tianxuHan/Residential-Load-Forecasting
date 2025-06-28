import numpy as np
from TEcal import transfer_entropy
import pandas as pd
from math import sqrt


def generate_te(rd_user, source, save_path):
    adj = np.zeros((114, 114))
    # for i in range(32, len(rd_user)):
    for i in range(0, 114):
        adjsave = np.zeros((1, 114))
        # for j in range(0, len(rd_user)):0
        for j in range(0, 114):
            X = np.loadtxt(source, dtype=np.float16, usecols=i, delimiter=",")
            Y = np.loadtxt(source, dtype=np.float16, usecols=j, delimiter=",")
            Y = np.array(Y)
            X = np.array(X)
            X = [sum(X[i:i + 60]) for i in range(0, len(X), 60)]
            Y = [sum(Y[i:i + 60]) for i in range(0, len(Y), 60)]
            adj[i][j] = transfer_entropy(X, Y, delay=1)
            print(i, j,adj[i][j])
            adjsave[0][j] = adj[i][j]
        adjsave = np.array(adjsave)
        adjsave = pd.DataFrame(adjsave)
        adjsave.to_csv(save_path, index=False,mode="a+", header=False)
    # adj = np.array(adj)
    # adj = pd.DataFrame(adj)
    # adj.to_csv(save_path, index=False, header=False)
    print("te complete")
    return adj


def generate_prs(rd_user, source, save_path):
    adj = np.zeros((114,114))
    for i in range(0, 114):
        adjsave = np.zeros((1, 114))
        for j in range(0, 114):
            X = np.loadtxt(source, dtype=np.float, usecols=i, delimiter=",")
            Y = np.loadtxt(source, dtype=np.float, usecols=j, delimiter=",")
            Y = np.array(Y)
            X = np.array(X)
            adj[i][j] = corrcoef(X, Y)
            print(i, j)
            adjsave[0][j] = adj[i][j]
        adjsave = np.array(adjsave)
        adjsave = pd.DataFrame(adjsave)
        adjsave.to_csv(save_path, index=False, mode="a+",header=False)
        print("prs complete")
    return adj


def multipl(a, b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab


def corrcoef(x, y):
    n = len(x)
    sum1 = sum(x)
    sum2 = sum(y)
    sumofxy = multipl(x, y)
    sumofx2 = sum([pow(i, 2) for i in x])
    sumofy2 = sum([pow(j, 2) for j in y])
    num = sumofxy - (float(sum1) * float(sum2) / n)
    den = sqrt((sumofx2 - float(sum1 ** 2) / n) * (sumofy2 - float(sum2 ** 2) / n))
    return num / den
