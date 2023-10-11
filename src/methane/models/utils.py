import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        query = self.query(x).view(x.size(0), -1, x.size(2) * x.size(3)).permute(0, 2, 1)
        key = self.key(x).view(x.size(0), -1, x.size(2) * x.size(3))
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value(x).view(x.size(0), -1, x.size(2) * x.size(3))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(x.size(0), x.size(1), x.size(2), x.size(3))
        out = self.gamma * out + x
        return out