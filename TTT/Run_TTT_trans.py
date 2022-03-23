import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import TTT.Data_TTT
from TTT.Data_TTT import Bs_exp_train, Bs_exp_test, X_Bs_exp_train, X_Bs_exp_test, y_Bs_exp_train, y_Bs_exp_test
from scipy.stats import pearsonr

# 超参数
LR = 0.2
EPOCH = 20

# 模型
net = torch.load('TTTsourcemodel.pth')
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LR)

# 训练 测试
for epoch in range(EPOCH):
    net.train()

    for i, (inputs, labels) in enumerate(Bs_exp_train):
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Epoch [%d/%d], Step[%d/%d], Loss: %.4f' % (epoch + 1, EPOCH, i + 1,
                                                          (len(X_Bs_exp_train) // TTT.Data_TTT.BATCHSIZE) + 1, loss.item()))

    net.eval()

    for epoch2 in range(1):
        for i, (inputs, labels) in enumerate(Bs_exp_test):
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            print('Test set: Epoch [%d/%d], Loss: %.4f' % (epoch + 1, EPOCH, loss.item()))

    pred_train = net(torch.tensor(X_Bs_exp_train)).detach().numpy()
    pred_test = net(torch.tensor(X_Bs_exp_test)).detach().numpy()

    pred_train = np.squeeze(pred_train)
    pred_test = np.squeeze(pred_test)
    true_train = np.squeeze(y_Bs_exp_train)
    true_test = np.squeeze(y_Bs_exp_test)

    cc_train = pearsonr(pred_train, true_train)
    cc_test = pearsonr(pred_test, true_test)

    print("train：", cc_train)
    print("test：", cc_test)
    print()
