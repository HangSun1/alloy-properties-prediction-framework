import pandas as pd
import numpy as np
import torch
import sklearn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 超参数
BATCHSIZE = 16

np.set_printoptions(suppress=True)
CCT_sim_data = pd.read_csv(r'CCT_sim.csv', dtype='float32')
CCT_exp_data = pd.read_csv(r'CCT_exp.csv', dtype='float32')

CCT_sim_data = pd.DataFrame(CCT_sim_data)
CCT_exp_data = pd.DataFrame(CCT_exp_data)

X_CCT_sim = CCT_sim_data.iloc[:, 0:18]
y_CCT_sim = CCT_sim_data.iloc[:, 18]
X_CCT_sim_train, X_CCT_sim_test, y_CCT_sim_train, y_CCT_sim_test = train_test_split(X_CCT_sim, y_CCT_sim, test_size=0.1)
y_CCT_sim_train = y_CCT_sim_train.to_frame()
y_CCT_sim_test = y_CCT_sim_test.to_frame()

X_CCT_exp = CCT_exp_data.iloc[:, 0:18]
y_CCT_exp = CCT_exp_data.iloc[:, 18]
X_CCT_exp_train, X_CCT_exp_test, y_CCT_exp_train, y_CCT_exp_test = train_test_split(X_CCT_exp, y_CCT_exp, test_size=0.1)
y_CCT_exp_train = y_CCT_exp_train.to_frame()
y_CCT_exp_test = y_CCT_exp_test.to_frame()

# 归一化
# 仿真数据
CCT_norm_x_sim = sklearn.preprocessing.MinMaxScaler()
X_CCT_sim_train = CCT_norm_x_sim.fit_transform(X_CCT_sim_train)
X_CCT_sim_test = CCT_norm_x_sim.transform(X_CCT_sim_test)

CCT_norm_y_sim = sklearn.preprocessing.MinMaxScaler()
y_CCT_sim_train = CCT_norm_y_sim.fit_transform(y_CCT_sim_train)
y_CCT_sim_test = CCT_norm_y_sim.transform(y_CCT_sim_test)

# 实验数据
CCT_norm_x_exp = sklearn.preprocessing.MinMaxScaler()
X_CCT_exp_train = CCT_norm_x_exp.fit_transform(X_CCT_exp_train)
X_CCT_exp_test = CCT_norm_x_exp.transform(X_CCT_exp_test)

CCT_norm_y_exp = sklearn.preprocessing.MinMaxScaler()
y_CCT_exp_train = CCT_norm_y_exp.fit_transform(y_CCT_exp_train)
y_CCT_exp_test = CCT_norm_y_exp.transform(y_CCT_exp_test)

# dataloader
CCT_sim_train = DataLoader(TensorDataset(torch.tensor(X_CCT_sim_train), torch.tensor(y_CCT_sim_train)),
                           shuffle=True, batch_size=BATCHSIZE)
CCT_sim_test = DataLoader(TensorDataset(torch.tensor(X_CCT_sim_test), torch.tensor(y_CCT_sim_test)),
                        shuffle=False, batch_size=10000)

CCT_exp_train = DataLoader(TensorDataset(torch.tensor(X_CCT_exp_train), torch.tensor(y_CCT_exp_train)),
                         shuffle=True, batch_size=BATCHSIZE)
CCT_exp_test = DataLoader(TensorDataset(torch.tensor(X_CCT_exp_test), torch.tensor(y_CCT_exp_test)),
                        shuffle=False, batch_size=BATCHSIZE)
