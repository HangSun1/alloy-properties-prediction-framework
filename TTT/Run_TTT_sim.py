import TTT.Data_TTT
from TTT.Data_TTT import Bs_sim_train, Bs_sim_test, X_Bs_sim_train, X_Bs_sim_test, y_Bs_sim_train, y_Bs_sim_test, Bs_norm_y_sim
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from TTT import Model

# 超参数
LR = 0.2
EPOCH = 100

# 模型
net = Model.source_net()
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LR)

# 训练 测试
for epoch in range(EPOCH):
    net.train()

    for i, (inputs, labels) in enumerate(Bs_sim_train):
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Epoch [%d/%d], Step[%d/%d], Loss: %.4f' % (epoch + 1, EPOCH, i + 1,
                                                          (len(X_Bs_sim_train) // TTT.Data_TTT.BATCHSIZE) + 1, loss.item()))

    net.eval()

    for epoch2 in range(1):
        for i, (inputs, labels) in enumerate(Bs_sim_test):
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            print('Test set: Epoch [%d/%d], Loss: %.4f' % (epoch + 1, EPOCH, loss.item()))

    pred_train = net(torch.tensor(X_Bs_sim_train)).detach().numpy()
    pred_test = net(torch.tensor(X_Bs_sim_test)).detach().numpy()

    pred_train = np.squeeze(pred_train)
    pred_test = np.squeeze(pred_test)
    true_train = np.squeeze(y_Bs_sim_train)
    true_test = np.squeeze(y_Bs_sim_test)

    cc_train = pearsonr(pred_train, true_train)
    cc_test = pearsonr(pred_test, true_test)

    print("train：", cc_train)
    print("test：", cc_test)
    print()

torch.save(net, r'E:\code\code\TTT\TTTsourcemodel.pth')
