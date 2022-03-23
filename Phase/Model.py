import torch
import torch.nn as nn
import torch.nn.functional as F

class source_net(nn.Module):
    def __init__(self):
        super(source_net, self).__init__()
        self.Linear1 = nn.Linear(15, 1024)
        self.Linear2 = nn.Linear(1024, 512)
        self.Linear3 = nn.Linear(512, 256)
        self.Linear4 = nn.Linear(256, 128)
        self.Linear5 = nn.Linear(128, 64)
        self.Linear6 = nn.Linear(64, 32)
        self.Linear7 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.Linear1(x))
        x = F.dropout(x, p=0.1)
        x = F.relu(self.Linear2(x))
        x = F.dropout(x, p=0.1)
        x = F.relu(self.Linear3(x))
        x = F.dropout(x, p=0.1)
        x = F.relu(self.Linear4(x))
        x = F.dropout(x, p=0.1)
        x = F.relu(self.Linear5(x))
        x = F.relu(self.Linear6(x))
        x = F.softmax(self.Linear7(x), dim=1)

        return x
