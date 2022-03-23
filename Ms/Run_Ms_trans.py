import Ms.Data_Ms
from Ms.Data_Ms import Ms_exp_train, Ms_exp_test, X_Ms_exp_train, X_Ms_exp_test, y_Ms_exp_train, y_Ms_exp_test, Ms_norm_y_exp
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 超参数
LR = 0.3
EPOCH = 20

# 模型
net = torch.load('Mssourcemodel.pth')
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LR)

# 训练 测试
for epoch in range(EPOCH):
    net.train()

    for i, (inputs, labels) in enumerate(Ms_exp_train):
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Epoch[%d/%d], Step[%d/%d], Loss: %.4f' % (epoch + 1, EPOCH, i + 1,
                                                         (len(X_Ms_exp_train) // Ms.Data_Ms.BATCHSIZE) + 1, loss.item()))

    net.eval()

    for epoch2 in range(1):
        for i, (inputs, labels) in enumerate(Ms_exp_test):
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            print('Test set: Epoch [%d/%d], Loss: %.4f' % (epoch + 1, EPOCH, loss.item()))

    pred_train = net(torch.tensor(X_Ms_exp_train)).detach().numpy()
    pred_test = net(torch.tensor(X_Ms_exp_test)).detach().numpy()

    pred_train = np.squeeze(pred_train)
    pred_test = np.squeeze(pred_test)
    true_train = np.squeeze(y_Ms_exp_train)
    true_test = np.squeeze(y_Ms_exp_test)

    cc_train = pearsonr(pred_train, true_train)
    cc_test = pearsonr(pred_test, true_test)

    print("train:", cc_train)
    print("test:", cc_test)
