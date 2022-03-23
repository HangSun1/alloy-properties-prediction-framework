from CCT import Data_CCT
from CCT.Data_CCT import CCT_sim_train, CCT_sim_test, X_CCT_sim_train, X_CCT_sim_test, y_CCT_sim_train, y_CCT_sim_test, CCT_norm_y_sim
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from CCT import Model
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

    for i, (inputs, labels) in enumerate(CCT_sim_train):
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Epoch[%d/%d], Step[%d/%d], Loss: %.4f' % (epoch + 1, EPOCH, i + 1,
                                                         (len(X_CCT_sim_train) // Data_CCT.BATCHSIZE) + 1, loss.item()))

    net.eval()

    for epoch2 in range(1):
        for i, (inputs, labels) in enumerate(CCT_sim_test):
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            print('Test set: Epoch [%d/%d], Loss: %.4f' % (epoch + 1, EPOCH, loss.item()))

    pred_train = net(torch.tensor(X_CCT_sim_train)).detach().numpy()
    pred_test = net(torch.tensor(X_CCT_sim_test)).detach().numpy()

    pred_train = np.squeeze(pred_train)
    pred_test = np.squeeze(pred_test)
    true_train = np.squeeze(y_CCT_sim_train)
    true_test = np.squeeze(y_CCT_sim_test)

    cc_train = pearsonr(pred_train, true_train)
    cc_test = pearsonr(pred_test, true_test)

    print("train:", cc_train)
    print("test:", cc_test)

torch.save(net, r'E:\code\code\CCT\CCTsourcemodel.pth')

plt.scatter(true_test, pred_test)
plt.xlabel('CCT simulation value')
plt.ylabel('CCT prediction value')
plt.grid(b=True)
plt.plot([0, 1], [0, 1], "k:")
plt.show()

