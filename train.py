import numpy as np
from torch_geometric.nn import Node2Vec
import os.path as osp
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from tqdm.notebook import tqdm

from Model import GCN_Encoder, MLPCluster, TransformerEncoder
from utility import graph_adj, best_map


def N2V_train():
    N2V.train()
    total_loss = 0
    for pos_rw, neg_rw in N2V_loader:
        N2V_optimizer.zero_grad()
        loss = N2V.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        N2V_optimizer.step()
        total_loss += loss.item()
    return total_loss / len(N2V_loader)


def T_train(co_train=False, detail=False):
    Transformer_encoder.train()
    Transformer_cluster.train()
    TC_optimizer.zero_grad()

    z, z_sim = Transformer_encoder(data.x, mask=diag_mask)
    c, c_sim = Transformer_cluster(z.squeeze(0), diag_mask)

    loss_embed = Transformer_encoder.recons_loss(z_sim, adj)
    loss_sim = Transformer_cluster.recons_loss(c_sim, adj)
    loss_mod = Transformer_cluster.modul_loss(c, adj)
    reg = Transformer_cluster.reg_term(c, n_nodes, n_class)
    if co_train:
        loss_clus = Transformer_cluster.soft_cross_entropy(c.log(), y)
        loss = 1 * loss_embed + 1 * loss_sim + 0.0001 * loss_clus + 0.2 * loss_mod
    else:
        loss = 0.7 * loss_embed + 1 * loss_sim + 0.1 * loss_mod + 0.2 * reg

    loss.backward()
    TC_optimizer.step()
    if detail:
        print('\nEpoch:', epoch, 'full_loss: {:.5f}'.format(loss.item()),
            'encoder_loss: {:.5f}'.format(loss_embed.item()), end=' ')

    return loss


def G_train(co_train=False, detail=False):
    Encoder.train()
    GCN_Cluster.train()
    GC_optimizer.zero_grad()

    x, x_sim = Encoder(n2v_embed, data.edge_index, diag_mask)
    c, c_sim = GCN_Cluster(x, diag_mask)
    loss_embed = Encoder.recons_loss(x_sim, adj)
    loss_sim = GCN_Cluster.recons_loss(c_sim, adj)
    loss_mod = GCN_Cluster.modul_loss(c, adj)
    reg = GCN_Cluster.reg_term(c, n_nodes, n_class)
    if co_train == True:
        loss_clus = GCN_Cluster.soft_cross_entropy(c.log(), y)
        loss = 1 * loss_embed + 1 * loss_sim + 0.0001 * loss_clus + 0.2 * loss_mod
    #         loss = 1*loss_sim + 1*loss_clus + 1*loss_mod
    else:
        loss = 1 * loss_embed + 1 * loss_sim + 0.1 * loss_mod + 0.5 * reg
    loss.backward()
    GC_optimizer.step()
    if detail:
        print('\nEpoch:', epoch, 'full_loss: {:.5f}'.format(loss.item()),
            'encoder_loss: {:.5f}'.format(loss_embed.item()), end=' ')

    return loss

def evalate():
    y_pred = torch.argmax(c, dim=1).cpu().detach().numpy()
    y_true = data.y.cpu().numpy()
    y_pred = best_map(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    print("\n\nAc:", acc, "NMI:", nmi, "F1:", f1_macro)
    return [acc, nmi, f1_macro]


if __name__ == '__main__':

    # Data Loading
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = 'Cora'

    MODEL_PATH = osp.join(osp.expanduser('~'), 'PycharmProjects\CoTrain\model', dataset)

    path = osp.join(osp.expanduser('~'), 'PycharmProjects\CoTrain\data', dataset)
    dataset = Planetoid(path, dataset)
    data = dataset[0]
    data.to(device)
    n_nodes = data.x.shape[0]
    n_class = max(data.y).item() + 1
    n_features = data.x.shape[1]

    adj = graph_adj(n_nodes, data.edge_index).to(device)
    diag_mask = torch.eye(n_nodes, n_nodes, dtype=torch.bool).to(device)

    # Hyper-params
    emb_dim = 256

    # Node2Vec Param
    N2V_Epochs = 100
    walk_length = 15
    context_size = 10
    walks_per_node = 5
    num_negative_samples = 1
    p = 5
    q = 7

    # GCN Param
    GCN_Epochs = 300
    out_channels = 126

    # Transformer Param
    T_Epochs = 500
    raw_dim = n_features
    input_dim = 256
    num_layers = 1
    num_heads = 4

    # Cluster Param
    hid_dim = 64
    co_epoch = 300

    res = []
    res_init = []
    res_GNN =[]

    # Node2Vec Train
    print('_______________________________')
    print('__Training Node2Vec__')
    N2V = Node2Vec(data.edge_index, embedding_dim=emb_dim, walk_length=walk_length,
                   context_size=context_size, walks_per_node=walks_per_node,
                   num_negative_samples=num_negative_samples, p=p, q=q, sparse=True).to(device)
    N2V_loader = N2V.loader(batch_size=128, shuffle=True, num_workers=0)
    N2V_optimizer = torch.optim.SparseAdam(list(N2V.parameters()), lr=0.01)

    for epoch in range(N2V_Epochs):
        loss = N2V_train()
        # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    n2v_embed = N2V()  # Node2Vec Embedding

    for _ in range(10):

        # Transformer Train
        print('__Training Transformer__')
        Transformer_encoder = TransformerEncoder(raw_dim=raw_dim, input_dim=input_dim,
                                                 num_layers=num_layers, num_heads=num_heads,
                                                 dim_feedforward=4 * input_dim).to(device)

        Transformer_cluster = MLPCluster(input_dim=input_dim, hid_dim=hid_dim, num_comm=n_class).to(device)

        TC_optimizer = torch.optim.Adam((list(Transformer_encoder.parameters()) + list(Transformer_cluster.parameters())),
                                        lr=0.001, weight_decay=0)

        for p in Transformer_encoder.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        best_loss = 1e9
        for epoch in range(300):
            loss = T_train()

            if loss.item() < best_loss:
                bad_count = 0
                best_loss = loss.item()
                torch.save(Transformer_encoder.state_dict(), MODEL_PATH + "encoder")
                torch.save(Transformer_cluster.state_dict(), MODEL_PATH + "clustermodel")

            z, z_sim = Transformer_encoder(data.x, mask=diag_mask)
            c, c_sim = Transformer_cluster(z.squeeze(0), diag_mask)

            if epoch % 50 == 0:
                _ = evalate()

        print("__ Evaluate Transformer__")
        Transformer_encoder.load_state_dict(torch.load(MODEL_PATH + "encoder"))
        Transformer_cluster.load_state_dict(torch.load(MODEL_PATH + "clustermodel"))
        Transformer_encoder.eval()
        Transformer_cluster.eval()
        z, z_sim = Transformer_encoder(data.x, mask=diag_mask)
        c, _ = Transformer_cluster(z.squeeze(0), diag_mask)

        t1_out = evalate()
        res_init.append(t1_out)

        # Co-training
        for c_epoch in range(5):
            print ('Co-train round: ', c_epoch)
            z, z_sim = Transformer_encoder(data.x, mask=diag_mask)
            c, _ = Transformer_cluster(z.squeeze(0), diag_mask)
            y = c.detach()

            Encoder = GCN_Encoder(in_channels=emb_dim, out_channels=out_channels, k=3).to(device)
            GCN_Cluster = MLPCluster(input_dim=out_channels, hid_dim=hid_dim, num_comm=n_class).to(device)
            GC_optimizer = torch.optim.Adam((list(Encoder.parameters()) + list(GCN_Cluster.parameters())), lr=0.001)

            best_loss = 1e9
            for epoch in range(100):
                loss = G_train(co_train=True)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(Encoder.state_dict(), MODEL_PATH + "g_encoder")
                    torch.save(GCN_Cluster.state_dict(), MODEL_PATH + "g_clustermodel")

                x, x_sim = Encoder(n2v_embed, data.edge_index, diag_mask)
                c, c_sim = GCN_Cluster(x, diag_mask)

                if epoch % 50 == 0:
                    y_pred = torch.argmax(c, dim=1).cpu().detach().numpy()
                    y_true = data.y.cpu().numpy()
                    y_pred = best_map(y_true, y_pred)
                    acc = metrics.accuracy_score(y_true, y_pred)
                    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
                    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
                    print("\n\nAc:", acc, "NMI:", nmi, "F1:", f1_macro)

            Encoder.load_state_dict(torch.load(MODEL_PATH + "g_encoder"))
            GCN_Cluster.load_state_dict(torch.load(MODEL_PATH + "g_clustermodel"))
            Encoder.eval()
            GCN_Cluster.eval()
            x, x_sim = Encoder(n2v_embed, data.edge_index, diag_mask)
            c, _ = GCN_Cluster(x, diag_mask)
            print("__ GNN Evaluation__")
            gnn_eval = evalate()
            y = c.detach()

            Transformer_encoder = TransformerEncoder(raw_dim=raw_dim, input_dim=input_dim,
                                                     num_layers=num_layers, num_heads=num_heads,
                                                     dim_feedforward=4 * input_dim).to(device)

            Transformer_cluster = MLPCluster(input_dim=input_dim, hid_dim=hid_dim, num_comm=n_class).to(device)

            TC_optimizer = torch.optim.Adam((list(Transformer_encoder.parameters()) + list(Transformer_cluster.parameters())),
                                            lr=0.001, weight_decay=0)
            best_loss = 1e9
            for epoch in range(200):
                loss = T_train(co_train=True)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(Transformer_encoder.state_dict(), MODEL_PATH + "encoder")
                    torch.save(Transformer_cluster.state_dict(), MODEL_PATH + "clustermodel")

                z, z_sim = Transformer_encoder(data.x, mask=diag_mask)
                c, c_sim = Transformer_cluster(z.squeeze(0), diag_mask)

                if epoch % 50 == 0:
                    _ = evalate()

        print("__ Final Evaluation__")
        Transformer_encoder.load_state_dict(torch.load(MODEL_PATH + "encoder"))
        Transformer_cluster.load_state_dict(torch.load(MODEL_PATH + "clustermodel"))
        Transformer_encoder.eval()
        Transformer_cluster.eval()
        z, z_sim = Transformer_encoder(data.x, mask=diag_mask)
        c, _ = Transformer_cluster(z.squeeze(0), diag_mask)

        final = evalate()
        res.append(final)
        res_GNN.append(gnn_eval)

    res = np.array(res)
    print(res)
    np.savetxt("cotrain_cora_avg7.csv", res, delimiter=",")
    res_init = np.array(res_init)
    res_GNN = np.array(res_GNN)
    np.savetxt("cotrain_cora_init7.csv", res_init, delimiter=",")
    np.savetxt("cotrain_cora_GNN7.csv", res_GNN, delimiter=",")

