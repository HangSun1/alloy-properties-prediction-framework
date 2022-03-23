import pandas as pd
import numpy as np
import torch
import sklearn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 超参数
BATCHSIZE = 16

np.set_printoptions(suppress=True)
Phase_sim_data = pd.read_csv(r'Phase_sim.csv', dtype='float32')
Phase_exp_data = pd.read_csv(r'Phase_exp.csv', dtype='float32')

Phase_sim_data = pd.DataFrame(Phase_sim_data)
Phase_exp_data = pd.DataFrame(Phase_exp_data)

X_Phase_sim = Phase_sim_data.iloc[:, 0:15]
y_Phase_sim = Phase_sim_data.iloc[:, 15]
X_Phase_sim_train, X_Phase_sim_test, y_Phase_sim_train, y_Phase_sim_test = train_test_split(X_Phase_sim, y_Phase_sim, test_size=0.1)

X_Phase_exp = Phase_exp_data.iloc[:, 0:15]
y_Phase_exp = Phase_exp_data.iloc[:, 15]
X_Phase_exp_train, X_Phase_exp_test, y_Phase_exp_train, y_Phase_exp_test = train_test_split(X_Phase_exp, y_Phase_exp, test_size=0.1)

# 归一化
# 仿真数据
Phase_norm_x_sim = sklearn.preprocessing.MinMaxScaler()
X_Phase_sim_train = Phase_norm_x_sim.fit_transform(X_Phase_sim_train)
X_Phase_sim_test = Phase_norm_x_sim.transform(X_Phase_sim_test)

# 实验数据
Phase_norm_x_exp = sklearn.preprocessing.MinMaxScaler()
X_Phase_exp_train = Phase_norm_x_exp.fit_transform(X_Phase_exp_train)
X_Phase_exp_test = Phase_norm_x_exp.transform(X_Phase_exp_test)

# dataloader
Phase_sim_train = DataLoader(TensorDataset(torch.tensor(X_Phase_sim_train).float(),
                                           torch.tensor(y_Phase_sim_train.values).float()),
                             shuffle=True, batch_size=BATCHSIZE)
Phase_sim_test = DataLoader(TensorDataset(torch.tensor(X_Phase_sim_test).float(),
                                          torch.tensor(y_Phase_sim_test.values).float()),
                            shuffle=False, batch_size=10000)

Phase_exp_train = DataLoader(TensorDataset(torch.tensor(X_Phase_exp_train).float(),
                                           torch.tensor(y_Phase_exp_train.values).float()),
                             shuffle=True, batch_size=BATCHSIZE)
Phase_exp_test = DataLoader(TensorDataset(torch.tensor(X_Phase_exp_test).float(),
                                          torch.tensor(y_Phase_exp_test.values).float()),
                            shuffle=False, batch_size=BATCHSIZE)
