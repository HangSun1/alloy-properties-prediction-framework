import pandas as pd
import numpy as np
import torch
import sklearn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 超参数
BATCHSIZE = 16

np.set_printoptions(suppress=True)
Ms_sim_data = pd.read_csv(r'Ms_sim.csv', dtype='float32')
Ms_exp_data = pd.read_csv(r'Ms_exp.csv', dtype='float32')

Ms_sim_data = pd.DataFrame(Ms_sim_data)
Ms_exp_data = pd.DataFrame(Ms_exp_data)

X_Ms_sim = Ms_sim_data.iloc[:, 0:17]
y_Ms_sim = Ms_sim_data.iloc[:, 17]
# print(y_Ms_sim)
X_Ms_sim_train, X_Ms_sim_test, y_Ms_sim_train, y_Ms_sim_test = train_test_split(X_Ms_sim, y_Ms_sim, test_size=0.1)
y_Ms_sim_train = y_Ms_sim_train.to_frame()
y_Ms_sim_test = y_Ms_sim_test.to_frame()

X_Ms_exp = Ms_exp_data.iloc[:, 0:17]
y_Ms_exp = Ms_exp_data.iloc[:, 17]
# print(y_Ms_sim)
X_Ms_exp_train, X_Ms_exp_test, y_Ms_exp_train, y_Ms_exp_test = train_test_split(X_Ms_exp, y_Ms_exp, test_size=0.1)
y_Ms_exp_train = y_Ms_exp_train.to_frame()
y_Ms_exp_test = y_Ms_exp_test.to_frame()

# 归一化
# 仿真数据
Ms_norm_x_sim = sklearn.preprocessing.MinMaxScaler()
X_Ms_sim_train = Ms_norm_x_sim.fit_transform(X_Ms_sim_train)
X_Ms_sim_test = Ms_norm_x_sim.transform(X_Ms_sim_test)

Ms_norm_y_sim = sklearn.preprocessing.MinMaxScaler()
y_Ms_sim_train = Ms_norm_y_sim.fit_transform(y_Ms_sim_train)
y_Ms_sim_test = Ms_norm_y_sim.transform(y_Ms_sim_test)

# 实验数据
Ms_norm_x_exp = sklearn.preprocessing.MinMaxScaler()
X_Ms_exp_train = Ms_norm_x_exp.fit_transform(X_Ms_exp_train)
X_Ms_exp_test = Ms_norm_x_exp.transform(X_Ms_exp_test)

Ms_norm_y_exp = sklearn.preprocessing.MinMaxScaler()
y_Ms_exp_train = Ms_norm_y_exp.fit_transform(y_Ms_exp_train)
y_Ms_exp_test = Ms_norm_y_exp.transform(y_Ms_exp_test)

# dataloader
Ms_sim_train = DataLoader(TensorDataset(torch.tensor(X_Ms_sim_train), torch.tensor(y_Ms_sim_train)),
                          shuffle=True, batch_size=BATCHSIZE)
Ms_sim_test = DataLoader(TensorDataset(torch.tensor(X_Ms_sim_test), torch.tensor(y_Ms_sim_test)),
                         shuffle=False, batch_size=10000)

Ms_exp_train = DataLoader(TensorDataset(torch.tensor(X_Ms_exp_train), torch.tensor(y_Ms_exp_train)),
                          shuffle=True, batch_size=BATCHSIZE)
Ms_exp_test = DataLoader(TensorDataset(torch.tensor(X_Ms_exp_test), torch.tensor(y_Ms_exp_test)),
                         shuffle=False, batch_size=BATCHSIZE)
