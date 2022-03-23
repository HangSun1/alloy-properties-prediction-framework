from E import Data_E
from E.Data_E import E_sim_train, E_sim_test, X_E_sim_train, X_E_sim_test, y_E_sim_train, y_E_sim_test, E_norm_y_sim
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from E import Model
import matplotlib.pyplot as plt

# 超参数
LR = 0.1
EPOCH = 100

# 模型
net = Model.source_net()
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LR)


# 训练 测试
for epoch in range(EPOCH):
    net.train()

    for i, (inputs, labels) in enumerate(E_sim_train):
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Epoch[%d/%d], Step[%d/%d], Loss: %.4f' % (epoch + 1, EPOCH, i + 1,
                                                         (len(X_E_sim_train) // Data_E.BATCHSIZE) + 1, loss.item()))

    net.eval()

    for epoch2 in range(1):
        for i, (inputs, labels) in enumerate(E_sim_test):
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            print('Test set: Epoch [%d/%d], Loss: %.4f' % (epoch + 1, EPOCH, loss.item()))

    pred_train = net(torch.tensor(X_E_sim_train)).detach().numpy()
    pred_test = net(torch.tensor(X_E_sim_test)).detach().numpy()

    pred_train = np.squeeze(pred_train)
    pred_test = np.squeeze(pred_test)
    true_train = np.squeeze(y_E_sim_train)
    true_test = np.squeeze(y_E_sim_test)

    cc_train = pearsonr(pred_train, true_train)
    cc_test = pearsonr(pred_test, true_test)

    print("train:", cc_train)
    print("test:", cc_test)

torch.save(net, r'E:\code\code\E\Esourcemodel.pth')

plt.scatter(true_test, pred_test)
plt.xlabel('E simulation value')
plt.ylabel('E prediction value')
plt.grid(b=True)
plt.plot([0, 1], [0, 1], "k:")
plt.show()

