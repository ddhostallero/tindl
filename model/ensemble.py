import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsModel(nn.Module):
    def __init__(self, model_list):
        super(EnsModel, self).__init__()
        self.model_list = model_list
        
    def forward(self, x):
        outs = []
        for model in self.model_list:
            out = model(x)
            outs.append(out)
        return torch.cat(outs, axis=1).mean(axis=1, keepdim=True)

    def predict_indiv(self, x):
        outs = []
        for model in self.model_list:
            out = model(x)
            outs.append(out)

        return torch.cat(outs, axis=1)