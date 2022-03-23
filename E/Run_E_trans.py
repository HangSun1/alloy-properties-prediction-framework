from E import Data_E
from E.Data_E import E_exp_train, E_exp_test, X_E_exp_train, X_E_exp_test, y_E_exp_train, y_E_exp_test, E_norm_y_exp
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 超参数
LR = 0.2
EPOCH = 100

# 模型
net = torch.load('Esourcemodel.pth')
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LR)

# 训练 测试
for epoch in range(EPOCH):
    net.train()

    for i, (inputs, labels) in enumerate(E_exp_train):
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Epoch[%d/%d], Step[%d/%d], Loss: %.4f' % (epoch + 1, EPOCH, i + 1,
                                                         (len(X_E_exp_train) // Data_E.BATCHSIZE) + 1, loss.item()))

    net.eval()

    for epoch2 in range(1):
        for i, (inputs, labels) in enumerate(E_exp_test):
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            print('Test set: Epoch [%d/%d], Loss: %.4f' % (epoch + 1, EPOCH, loss.item()))

    pred_train = net(torch.tensor(X_E_exp_train)).detach().numpy()
    pred_test = net(torch.tensor(X_E_exp_test)).detach().numpy()

    pred_train = np.squeeze(pred_train)
    pred_test = np.squeeze(pred_test)
    true_train = np.squeeze(y_E_exp_train)
    true_test = np.squeeze(y_E_exp_test)

    cc_train = pearsonr(pred_train, true_train)
    cc_test = pearsonr(pred_test, true_test)

    print("train:", cc_train)
    print("test:", cc_test)

plt.scatter(true_test, pred_test)
plt.xlabel('E experiment value')
plt.ylabel('E prediction value')
plt.grid(b=True)
plt.plot([0, 1], [0, 1], "k:")
plt.show()
