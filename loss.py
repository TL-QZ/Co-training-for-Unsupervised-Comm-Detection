import torch

def modularity_loss(c_dist, adj):
    degrees = adj.sum(0)
    w = adj.sum()
#     print(w)
    mod = torch.sum(torch.mul(c_dist, torch.matmul(adj, c_dist)))
#     print(mod)
    C_d = torch.matmul(degrees, c_dist)
    mod2 = torch.matmul(C_d, C_d.transpose(0, 1))
    mod = mod + mod2 / w
    mod /= w
    return -mod