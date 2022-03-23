import pandas as pd
import numpy as np
import torch
import sklearn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 超参数
BATCHSIZE = 16

np.set_printoptions(suppress=True)
E_sim_data = pd.read_csv(r'E_sim.csv', dtype='float32')
E_exp_data = pd.read_csv(r'E_exp.csv', dtype='float32')

E_sim_data = pd.DataFrame(E_sim_data)
E_exp_data = pd.DataFrame(E_exp_data)

X_E_sim = E_sim_data.iloc[:, 0:16]
y_E_sim = E_sim_data.iloc[:, 16]
X_E_sim_train, X_E_sim_test, y_E_sim_train, y_E_sim_test = train_test_split(X_E_sim, y_E_sim, test_size=0.1)
y_E_sim_train = y_E_sim_train.to_frame()
y_E_sim_test = y_E_sim_test.to_frame()

X_E_exp = E_exp_data.iloc[:, 0:16]
y_E_exp = E_exp_data.iloc[:, 16]
X_E_exp_train, X_E_exp_test, y_E_exp_train, y_E_exp_test = train_test_split(X_E_exp, y_E_exp, test_size=0.1)
y_E_exp_train = y_E_exp_train.to_frame()
y_E_exp_test = y_E_exp_test.to_frame()

# 归一化
# 仿真数据
E_norm_x_sim = sklearn.preprocessing.MinMaxScaler()
X_E_sim_train = E_norm_x_sim.fit_transform(X_E_sim_train)
X_E_sim_test = E_norm_x_sim.transform(X_E_sim_test)

E_norm_y_sim = sklearn.preprocessing.MinMaxScaler()
y_E_sim_train = E_norm_y_sim.fit_transform(y_E_sim_train)
y_E_sim_test = E_norm_y_sim.transform(y_E_sim_test)

# 实验数据
E_norm_x_exp = sklearn.preprocessing.MinMaxScaler()
X_E_exp_train = E_norm_x_exp.fit_transform(X_E_exp_train)
X_E_exp_test = E_norm_x_exp.transform(X_E_exp_test)

E_norm_y_exp = sklearn.preprocessing.MinMaxScaler()
y_E_exp_train = E_norm_y_exp.fit_transform(y_E_exp_train)
y_E_exp_test = E_norm_y_exp.transform(y_E_exp_test)

# dataloader
E_sim_train = DataLoader(TensorDataset(torch.tensor(X_E_sim_train), torch.tensor(y_E_sim_train)),
                         shuffle=True, batch_size=BATCHSIZE)
E_sim_test = DataLoader(TensorDataset(torch.tensor(X_E_sim_test), torch.tensor(y_E_sim_test)),
                        shuffle=False, batch_size=10000)

E_exp_train = DataLoader(TensorDataset(torch.tensor(X_E_exp_train), torch.tensor(y_E_exp_train)),
                         shuffle=True, batch_size=BATCHSIZE)
E_exp_test = DataLoader(TensorDataset(torch.tensor(X_E_exp_test), torch.tensor(y_E_exp_test)),
                        shuffle=False, batch_size=BATCHSIZE)
