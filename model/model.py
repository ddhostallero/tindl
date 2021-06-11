import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
  def __init__(self, n_in):
    super(FCN, self).__init__()
    
    self.l1 = nn.Linear(n_in, 512)    
    self.l2 = nn.Linear(512, 256)
    self.l3 = nn.Linear(256, 128)
    self.drp = nn.Dropout(0.5)
    self.l4 = nn.Linear(128, 1)

  def forward(self, x):
    x = F.relu(self.l1(x))
    x = F.relu(self.l2(x))
    x = F.relu(self.l3(x))
    x = self.drp(x)

    return self.l4(x)