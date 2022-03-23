import pandas as pd
import numpy as np
import torch
import sklearn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 超参数
BATCHSIZE = 16

np.set_printoptions(suppress=True)
Bs_sim_data = pd.read_csv(r'TTT_sim.csv', dtype='float32')
Bs_exp_data = pd.read_csv(r'TTT_exp.csv', dtype='float32')

Bs_sim_data = pd.DataFrame(Bs_sim_data)
Bs_exp_data = pd.DataFrame(Bs_exp_data)

Bs_sim_data['t'] = Bs_sim_data['t'].apply(np.log10)
Bs_exp_data['t'] = Bs_exp_data['t'].apply(np.log10)
# print(Bs_sim_data.dtypes)

X_Bs_sim = Bs_sim_data.iloc[:, 0:14]
y_Bs_sim = Bs_sim_data.iloc[:, 14]
X_Bs_sim_train, X_Bs_sim_test, y_Bs_sim_train, y_Bs_sim_test = train_test_split(X_Bs_sim, y_Bs_sim, test_size=0.1)
y_Bs_sim_train = y_Bs_sim_train.to_frame()  # 升维
# print(type(y_sim_test))
# print(type(y_sim_train))
y_Bs_sim_test = y_Bs_sim_test.to_frame()

X_Bs_exp = Bs_exp_data.iloc[:, 0:14]
y_Bs_exp = Bs_exp_data.iloc[:, 14]
X_Bs_exp_train, X_Bs_exp_test, y_Bs_exp_train, y_Bs_exp_test = train_test_split(X_Bs_exp, y_Bs_exp, test_size=0.1)
y_Bs_exp_train = y_Bs_exp_train.to_frame()
y_Bs_exp_test = y_Bs_exp_test.to_frame()

# 归一化
# 仿真数据
Bs_norm_x_sim = sklearn.preprocessing.MinMaxScaler()
X_Bs_sim_train = Bs_norm_x_sim.fit_transform(X_Bs_sim_train)
X_Bs_sim_test = Bs_norm_x_sim.transform(X_Bs_sim_test)
# print(type(X_Bs_sim_train))

Bs_norm_y_sim = sklearn.preprocessing.MinMaxScaler()
y_Bs_sim_train = Bs_norm_y_sim.fit_transform(y_Bs_sim_train)
y_Bs_sim_test = Bs_norm_y_sim.transform(y_Bs_sim_test)

# 实验数据
Bs_norm_x_exp = sklearn.preprocessing.MinMaxScaler()
X_Bs_exp_train = Bs_norm_x_exp.fit_transform(X_Bs_exp_train)
X_Bs_exp_test = Bs_norm_x_exp.transform(X_Bs_exp_test)

Bs_norm_y_exp = sklearn.preprocessing.MinMaxScaler()
y_Bs_exp_train = Bs_norm_y_exp.fit_transform(y_Bs_exp_train)
y_Bs_exp_test = Bs_norm_y_exp.transform(y_Bs_exp_test)

# 封装DataLoader
Bs_sim_train = DataLoader(TensorDataset(torch.tensor(X_Bs_sim_train), torch.tensor(y_Bs_sim_train)),
                          shuffle=True, batch_size=BATCHSIZE)
Bs_sim_test = DataLoader(TensorDataset(torch.tensor(X_Bs_sim_test), torch.tensor(y_Bs_sim_test)),
                         shuffle=False, batch_size=10000)

Bs_exp_train = DataLoader(TensorDataset(torch.tensor(X_Bs_exp_train), torch.tensor(y_Bs_exp_train)),
                          shuffle=True, batch_size=BATCHSIZE)
Bs_exp_test = DataLoader(TensorDataset(torch.tensor(X_Bs_exp_test), torch.tensor(y_Bs_exp_test)),
                         shuffle=False, batch_size=BATCHSIZE)
