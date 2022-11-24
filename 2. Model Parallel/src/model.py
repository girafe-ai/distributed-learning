import torch
import torch.nn as nn
import torch.nn.functional as F


# +
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1_0tp = nn.Linear(4608, 64)
        self.fc1_1tp = nn.Linear(4608, 64)
        self.fc2_0tp = nn.Linear(64, 64)
        self.fc2_1tp = nn.Linear(64, 64)
        
        self.to("cuda:0")
        self.fc1_1tp.to("cuda:1")
        self.fc2_1tp.to("cuda:1")
        
    def forward(self, x):
        x_tp1 = x.to("cuda:1")
        y_0 = F.relu(self.fc1_0tp(x))
        y_1 = F.relu(self.fc1_1tp(x_tp1))
        z_0 = F.relu(self.fc2_0tp(y_0))
        z_1 = F.relu(self.fc2_1tp(y_1))
        
        return torch.cat([z_0, z_1.to("cuda:0")], -1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_0tp = nn.Conv2d(1, 16, 3, 1)
        self.conv1_1tp = nn.Conv2d(1, 16, 3, 1)
        self.conv2_0tp = nn.Conv2d(32, 16, 3, 1)
        self.conv2_1tp = nn.Conv2d(32, 16, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        
        self.to("cuda:0")
        self.conv1_1tp.to("cuda:1")
        self.conv2_1tp.to("cuda:1")
        self.fc2.to("cuda:1")
        self.fc3.to("cuda:1")
        
    def forward(self, x):
        x_tp1 = x.to("cuda:1") # broadcast
        x_0 = F.relu(self.conv1_0tp(x))
        x_1 = F.relu(self.conv1_1tp(x_tp1))
        x = torch.cat([x_0, x_1.to("cuda:0")], 1) # all_gather
        x_tp1 = x.to("cuda:1")
        x_0 = F.relu(self.conv2_0tp(x))
        x_1 = F.relu(self.conv2_1tp(x_tp1))
        x = torch.cat([x_0, x_1.to("cuda:0")], 1)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = x.to("cuda:1")
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output.to("cuda:0")
