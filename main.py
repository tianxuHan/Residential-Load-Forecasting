import matplotlib.pyplot as plt
from spektral.layers import GCNConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, LSTM, Dense, Flatten, GRU
import numpy as np
import tensorflow as tf
import random
from numpy import *
import pandas as pd
import os
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from data_process import load_data, preprocess_data, threshold1, threshold2

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')


data, adj1_114, adj2_114 = load_data('fue')
adj1_114 = adj1_114 - adj1_114.T
time_len = data.shape[0]
num_nodes = data.shape[1]
data1 = np.mat(data, dtype=np.float32)

np.set_printoptions(threshold=np.inf)
adj1_114 = threshold1(0, adj1_114)
adj2_114 = threshold2(0, adj2_114)

ego_edge = 3
steps = 6
User_number = 45

ts=6


epochs = 100
ego_edge2 = 1

a = np.array(adj1_114[User_number, :])
a = a[0]
b = a.argsort()[-ego_edge:]
b = b[::-1]

c = np.array(adj2_114[User_number, :])
c = c[0]
d = c.argsort()[-ego_edge:]
d = d[::-1]

adj1_1140 = adj1_114
adj2_1140 = adj2_114
adj1_1140[:, User_number] = 0
adj2_1140[:, User_number] = 0

leaf1a = []
key = np.zeros((ego_edge - 1, ego_edge2))
for i in range(ego_edge - 1):
    adj1_1140[b[i + 1], b[i + 1]] = 0
    m = np.array(adj1_1140[b[i + 1], :])
    m = m[0]
    n = m.argsort()[-ego_edge2:]
    n = n[::-1]
    n = n.tolist()
    leaf1a.append(n)
    for j in range(ego_edge2):
        key[i][j] = n[j]

leaf1 = [i for item in leaf1a for i in item]
b = b.tolist()
b.extend(leaf1)
b = dict.fromkeys(b)
b = list(b.keys())

adj1 = np.zeros((len(b), len(b)))
for i in range(len(b)):
    adj1[i][i] = 1
for i in range(ego_edge - 1):
    adj1[0][i + 1] = adj1_114[User_number, b[i + 1]]
    # adj1[i+1][0] = adj2_114[b[i+1],User_number]
    adj1[i + 1][0] = adj1_114[User_number, b[i + 1]]

for i in range(ego_edge - 1):
    for j in range(ego_edge2):
        k = int(key[i][j])
        p = b.index(k)
        adj1[i + 1][p] = adj1_114[b[i + 1], k]
        adj1[p][i + 1] = adj1_114[b[i + 1], k]


# 第二张图
leaf2a = []
key2 = np.zeros((ego_edge - 1, ego_edge2))
for i in range(ego_edge - 1):
    adj2_1140[d[i + 1], d[i + 1]] = 0
    m = np.array(adj2_1140[d[i + 1], :])
    m = m[0]
    n = m.argsort()[-ego_edge2:]
    n = n[::-1]
    n = n.tolist()
    leaf2a.append(n)
    for j in range(ego_edge2):
        key2[i][j] = n[j]



leaf2 = [i for item in leaf2a for i in item]
d = d.tolist()
d.extend(leaf2)
d = dict.fromkeys(d)
d = list(d.keys())

adj2 = np.zeros((len(d), len(d)))
for i in range(len(d)):
    adj2[i][i] = 1
for i in range(ego_edge - 1):
    adj2[0][i + 1] = adj2_114[User_number, d[i + 1]]
    # adj1[i+1][0] = adj2_114[b[i+1],User_number]
    adj2[i + 1][0] = adj2_114[User_number, d[i + 1]]

for i in range(ego_edge - 1):
    for j in range(ego_edge2):
        k = int(key2[i][j])
        # print(k)
        # print(type(k))
        p = d.index(k)
        # print(p)
        # print(type(p))
        adj2[i + 1][p] = adj2_114[d[i + 1], k]
        adj2[p][i + 1] = adj2_114[d[i + 1], k]

datat = np.zeros((10080, len(b)))
datap = np.zeros((10080, len(d)))

for i in range(len(b)):
    t = data1[:, b[i]].flatten()
    datat[:, i] = t

for i in range(len(d)):
    t = data1[:, d[i]].flatten()
    datap[:, i] = t


datat = np.mat(datat, dtype=np.float32)
datap = np.mat(datap, dtype=np.float32)
max_value = np.max(datat)
min_value = np.min(datat)
datat = (datat - min_value) / (max_value - min_value)
datap = (datap - min_value) / (max_value - min_value)

trainX, trainY, valiX, valiY, testX, testY = preprocess_data(datat, time_len, 0.6, 0.2, ts, 1)
trainXp, trainYp, valiXp, valiYp, testXp, testYp = preprocess_data(datap, time_len, 0.6, 0.2, ts, 1)

trainX = tf.transpose(trainX, [0, 2, 1])
valiX = tf.transpose(valiX, [0, 2, 1])
testX = tf.transpose(testX, [0, 2, 1])
trainXp = tf.transpose(trainXp, [0, 2, 1])
valiXp = tf.transpose(valiXp, [0, 2, 1])
testXp = tf.transpose(testXp, [0, 2, 1])

valiX = np.array(valiX)
trainX = np.array(trainX)
testX = np.array(testX)
valiXp = np.array(valiXp)
trainXp = np.array(trainXp)
testXp = np.array(testXp)
trainY = trainY[:, -1]
testY = testY[:, -1]
valiY = valiY[:, -1]
trainY = trainY[:, 0]
testY = testY[:, 0]
valiY = valiY[:, 0]


adj1 = np.array(adj1)
adj2 = np.array(adj2)
ad1, ad2 = [], []
ad11, ad21 = [], []
ad12, ad22 = [], []

adj1 = GCNConv.preprocess(adj1)
adj2 = GCNConv.preprocess(adj2)

for i in range(int(time_len * 0.6) - ts - 1):
    ad1.append(adj1)
    ad2.append(adj2)
for i in range(int(time_len * 0.2) - ts-1 ):
    ad11.append(adj1)
    ad21.append(adj2)
for i in range(int(time_len * 0.2) - ts-1):
    ad12.append(adj1)
    ad22.append(adj2)

ad1 = np.array(ad1)
ad2 = np.array(ad2)
ad12 = np.array(ad12)
ad22 = np.array(ad22)
ad11 = np.array(ad11)
ad21 = np.array(ad21)


X_in = Input(shape=(len(b), ts))
X_in2 = Input(shape=(len(d), ts))
A_in2 = Input(shape=(len(d), len(d)))
A_in1 = Input(shape=(len(b), len(b)))

X_1 = GCNConv(64, activation='relu')([X_in, A_in1])
X_2 = GCNConv(32, 'relu')([X_1, A_in1])
x_2 = LSTM(100)(X_2)
Y_1 = GCNConv(64, activation='relu')([X_in2, A_in2])
Y_2 = GCNConv(32, 'relu')([Y_1, A_in2])
Y_2 = LSTM(100)(Y_2)
x = tf.keras.layers.Concatenate()([x_2,Y_2])
x = Dropout(0.2)(x)

x = Dense(1)(x)

model = Model(inputs=[X_in, X_in2, A_in1, A_in2], outputs=x)
model.summary()
model.compile(optimizer=keras.optimizers.Adam(0.001),  # adam优化器学习率0.001
              loss='mae', metrics=[tf.keras.metrics.RootMeanSquaredError()]
              )
history = model.fit([trainX, trainXp, ad1, ad2], trainY, validation_data=([valiX, valiXp, ad11, ad21], valiY),
                    batch_size=64, epochs=epochs)

history_dict = history.history
train_loss = history_dict['loss'] 

plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('loss.png')
plt.show()

predicted_load = model.predict([testX, testXp, ad12, ad22])
predicted_load = predicted_load * (max_value - min_value) + min_value
predicted_load[predicted_load < 0] = 0
var = pd.DataFrame(predicted_load)
var.to_csv(r'predicted load.csv', index=False, header=False)
real_load = testY * (max_value-min_value)+min_value
var2 = pd.DataFrame(real_load)
var2.to_csv(r'real load.csv', index=False, header=False)


rmse = math.sqrt(mean_squared_error(predicted_load, real_load))
mae = mean_absolute_error(predicted_load, real_load)
r2 = r2_score(predicted_load, real_load)
var = explained_variance_score(predicted_load, real_load)

print("-------------------------------------------------------------------")
print("Root Mean Square Error:" + str(rmse))
print("Mean Absolute Error:" + str(mae))
print("R2 score:" + str(r2))
print("Explained Variance Score:" + str(var))

plt.plot(real_load, color='red', label='real_load')
plt.plot(predicted_load, color='blue', label='Predicted load')
plt.title('load Prediction')
plt.xlabel('Time')
plt.ylabel('load')
plt.legend()
plt.savefig('curve.png')
plt.show()





