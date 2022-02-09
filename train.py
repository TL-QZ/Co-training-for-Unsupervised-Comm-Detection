import argparse
import math
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn import metrics
from yaml import SafeLoader

from layers.MLP_Clustering import MLPCluster
from loss import modularity_loss
from model import TransformerEncoder
from utility import data_process, best_map, get_dataset, load_wiki, load_synth_data


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Synth1')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--path', type=str, default='.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_arguments()
    cfg = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    MODEL_PATH = args.path + "/saved_models/" + args.dataset + "/"

    if args.dataset != "Wiki" and args.dataset != "Synth1":
        path = osp.join(osp.expanduser('~'), 'PycharmProjects\Transformer_Community_Detection\data', args.dataset)
        dataset = get_dataset(path, args.dataset)
        data = dataset[0]
        num_features = dataset.num_features
    elif args.dataset == "Synth1":
        path = osp.join(osp.expanduser('~'), 'PycharmProjects\Transformer_Community_Detection\data', args.dataset)
        data, num_features = load_synth_data(path)
    else:
        path = osp.join(osp.expanduser('~'), 'PycharmProjects\Transformer_Community_Detection\data')
        data, num_features = load_wiki(path)

    n_nodes = data.x.shape[0]
    n_class = max(data.y).item() + 1

    print("========", args.dataset, "========")
    print("Nodes    :", n_nodes)
    print("Features :", num_features)
    print("Classes  :", n_class)
    print("Distribution :", end=' ')
    for i in range(n_class-1):
        print(torch.sum(data.y == i).item(), end=', ')
    print(torch.sum(data.y == (n_class-1)).item(), end='\n')
    print("======================")

    items = data_process(data, device)
    diag_mask = torch.eye(n_nodes).to(device)

    res = []
    for _ in range(10):
        Transformer_encoder = TransformerEncoder(raw_dim=num_features, input_dim=cfg["num_hidden"],
                                                 num_layers=cfg["num_layers"],
                                                 num_heads=cfg["num_heads"],
                                                 dim_feedforward=4 * cfg["num_hidden"],
                                                 dropout=cfg["dropout"]).to(device)

        clustermodel = MLPCluster(input_dim=cfg["num_hidden"], hid_dim=cfg["cluster_hid_dim"], num_comm=n_class,
                                  dropout=cfg["cluster_dropout"]).to(device)
        fulloptimizer = torch.optim.Adam((list(Transformer_encoder.parameters()) + list(clustermodel.parameters())),
                                         lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

        show_loss = True
        best_loss = 1e9
        bad_count = 0
        patience = cfg["patience"]
        Epochs = cfg["epochs"]
        print("\n\n\nStarting final full training module")

        start = time.time()

        for epoch in range(Epochs + 1):
            Transformer_encoder.train()
            clustermodel.train()
            fulloptimizer.zero_grad()

            z, A_star = Transformer_encoder(items, mask=diag_mask)
            c_dist, pred_similarity = clustermodel(z, items.Adj)

            loss_embed = F.binary_cross_entropy(A_star, items.Adj)
            loss_reg = torch.linalg.norm(c_dist.sum(1)) / n_nodes * math.sqrt(n_class) - 1

            loss_sim = F.binary_cross_entropy(pred_similarity.squeeze(0), items.Adj)
            loss_mod = modularity_loss(c_dist, items.Adj)
            loss = 1.2 * loss_embed + 0.5 * loss_sim + 0.3 * loss_reg + 0.2 * loss_mod
            loss.backward()
            fulloptimizer.step()

            if show_loss:
                print('Epoch:', epoch, 'full_loss: {:.5f}'.format(loss.item()),
                      'encoder_loss: {:.5f}'.format(loss_embed.item()), end=' ')
                print('loss1: {:.5f}'.format(loss_sim.item()), 'loss2: {:.5f}'.format(loss_reg.item()),
                      'loss3: {:.5f}'.format(loss_mod.item()), flush=True)

            if loss.item() < best_loss:
                bad_count = 0
                best_loss = loss.item()
                torch.save(Transformer_encoder.state_dict(), MODEL_PATH + "encoder")
                torch.save(clustermodel.state_dict(), MODEL_PATH + "clustermodel")
            else:
                bad_count += 1
                print("Model not improved for", bad_count, "consecutive epochs..")
                if bad_count == patience:
                    print("Early stopping Cluster Train...")
                    break

            if epoch % 10 == 0:
                y_pred = torch.argmax(c_dist, dim=2).cpu().detach().numpy()
                y_true = data.y.cpu().numpy()
                y_pred = best_map(y_true, y_pred)
                acc = metrics.accuracy_score(y_true, y_pred[0])
                nmi = metrics.normalized_mutual_info_score(y_true, y_pred[0])
                f1_macro = metrics.f1_score(y_true, y_pred[0], average='macro')
                f1_micro = metrics.f1_score(y_true, y_pred[0], average='micro')
                print("\n\nAc:", acc, "NMI:", nmi, "F1Ma:", f1_macro, "F1Mi:", f1_micro)

        time_laps = time.time() - start
        print("Training time length", time_laps)
        Transformer_encoder.load_state_dict(torch.load(MODEL_PATH + "encoder"))
        clustermodel.load_state_dict(torch.load(MODEL_PATH + "clustermodel"))
        Transformer_encoder.eval()
        clustermodel.eval()
        z, A_star = Transformer_encoder(items)
        c_dist, pred_similarity = clustermodel(z, items.Adj)
        y_pred = torch.argmax(c_dist, dim=2).cpu().detach().numpy()
        y_true = data.y.cpu().numpy()
        y_pred = best_map(y_true, y_pred)
        acc = metrics.accuracy_score(y_true, y_pred[0])
        nmi = metrics.normalized_mutual_info_score(y_true, y_pred[0])
        f1_macro = metrics.f1_score(y_true, y_pred[0], average='macro')
        f1_micro = metrics.f1_score(y_true, y_pred[0], average='micro')
        print("\n\nAc:", acc, "NMI:", nmi, "F1Ma:", f1_macro, "F1Mi:", f1_micro)

        res.append([acc, nmi, f1_macro])

    res = np.array(res)
    print(res)
    np.savetxt("transformer_syn_avg.csv", res, delimiter=",")