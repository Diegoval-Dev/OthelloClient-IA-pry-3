# ==== policy_value_net.py ====
import torch
import torch.nn as nn
import torch.nn.functional as F

class OthelloNet(nn.Module):
    def __init__(self):
        super().__init__()
        # two input channels (white, black)
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 8 * 8, 64)
        # value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: [batch, 2, 8, 8]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # policy branch
        p = F.relu(self.policy_conv(x))          # [batch,2,8,8]
        p = p.view(p.size(0), -1)               # flatten to [batch, 128]
        policy_logits = self.policy_fc(p)       # [batch,64]

        # value branch
        v = F.relu(self.value_conv(x))          # [batch,1,8,8]
        v = v.view(v.size(0), -1)               # [batch,64]
        v = F.relu(self.value_fc1(v))           # [batch,64]
        value = torch.tanh(self.value_fc2(v))   # [batch,1]

        return policy_logits, value.squeeze(-1)

