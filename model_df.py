# model_def.py
import torch
import torch.nn as nn

class CodeViolationNet(nn.Module):
    """
    Same architecture you used during training:
    Input dim = 13, Hidden1 = 64, Hidden2 = 32, Output = 2.
    """
    def __init__(self, input_dim=13, hidden1=64, hidden2=32, output_dim=2):
        super(CodeViolationNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden2, output_dim)
        )

    def forward(self, x):
        return self.net(x)
