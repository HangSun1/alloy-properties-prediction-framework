import Phase.Data_Phase
from Phase.Data_Phase import Phase_exp_train, Phase_exp_test, X_Phase_exp_train, X_Phase_exp_test, y_Phase_exp_train, y_Phase_exp_test
import numpy as np
import torch
import torch.nn as nn

# 超参数
LR = 0.2
EPOCH = 90

# 模型
net = torch.load('Phasesourcemodel.pth')
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LR)

# 训练 测试
total = 0
correct = 0
for epoch in range(EPOCH):
    net.train()

    for i, (inputs, labels) in enumerate(Phase_exp_train):
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = loss_func(outputs, labels.long())
        loss.backward()
        optimizer.step()

        print('Epoch [%d/%d], Step[%d/%d], Loss: %.4f' % (epoch + 1, EPOCH, i + 1,
                                                          (len(X_Phase_exp_train) // Phase.Data_Phase.BATCHSIZE) + 1, loss.item()))

    net.eval()

    for epoch2 in range(1):
        for i, (inputs, labels) in enumerate(Phase_exp_test):
            outputs = net(inputs)
            loss = loss_func(outputs, labels.long())
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('Test set: Epoch [%d/%d], Loss: %.4f' % (epoch + 1, EPOCH, loss.item()))

    acc = correct / total

    print("accuracy：", acc)
    print()
