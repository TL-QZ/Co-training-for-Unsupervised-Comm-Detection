from torch.utils.data import Dataset


class AttributedGraph(Dataset):
    def __init__(self, x, y, Adj, device="cuda"):
        super(AttributedGraph, self).__init__()
        self.x = x.to(device)
        self.y = y.to(device)
        self.Adj = Adj.to(device)

    def __len__(self):
        return int(self.Adj.size()[0])

    def __getitem__(self, idx):
        return self.x[1, idx, :], self.y[idx], self.Adj[idx, idx]