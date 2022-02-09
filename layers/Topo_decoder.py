import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Topo_decoder(torch.nn.Module):
    def __init__(self, num_features, num_comm):
        super(Topo_decoder, self).__init__()
        self.weight = Parameter(torch.FloatTensor(num_comm, num_comm).uniform_(0, 0.01))
