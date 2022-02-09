import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPCluster(torch.nn.Module):
    def __init__(self, input_dim, hid_dim, num_comm, dropout=0.0):
        super(MLPCluster, self).__init__()
        self.mlp1 = nn.Linear(input_dim, hid_dim)
        self.mlp2 = nn.Linear(hid_dim, num_comm)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z, mask=None):
        assignment = F.elu(self.mlp1(z))
        assignment = self.mlp2(self.dropout(assignment))
        assignment = F.softmax(assignment, dim=-1)

        adjacency_logit = torch.matmul(assignment, torch.transpose(assignment, 1, 2))

        if mask is not None:
            adjacency_logit = adjacency_logit.masked_fill(mask == 0, 0)


        # if mask is not None:
        #     adjacency_logit = adjacency_logit.masked_fill(mask == 1, -9e15)
        #
        # adjacency = torch.sigmoid(adjacency_logit)

        return assignment, adjacency_logit
