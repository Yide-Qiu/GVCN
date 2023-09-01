
import argparse
import math
import time
import os
import shutil

import dgl
import numpy as np






import torch as th
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from outcome_correlation import prepare_folder

from models import GAT
import pickle as pkl
from tqdm import tqdm
import pdb


device = None
in_feats, n_classes = None, None
epsilon = 1 - math.log(2)


def gen_model(args):
    norm = "both" if args.use_norm else "none"

    if args.use_labels:
        model = GAT(
            in_feats + n_classes,
            n_classes,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            activation=F.relu,
            dropout=args.dropout,
            attn_drop=args.attn_drop,
            norm=norm,
        )
    else:
        model = GAT(
            in_feats,
            n_classes,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            activation=F.relu,
            dropout=args.dropout,
            attn_drop=args.attn_drop,
            norm=norm,
        )

    return model


def cross_entropy(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)

def focal_cross_entropy(x, labels):
    gama = 1
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    pt = th.exp(-y)
    y = ((1-pt)**gama*y).mean()
    return th.mean(y)    

def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred, "y_true": labels})["acc"]


def add_labels(feat, labels, idx):
    onehot = th.zeros([feat.shape[0], n_classes]).to(device)
    onehot[idx, labels[idx, 0]] = 1
    return th.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(model, graph, labels, train_idx, train_loader, optimizer, use_labels, focal_loss):
    model.train()

    feat = graph.ndata["feat"]

    if use_labels:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_pred_idx = train_idx[mask]

    total_loss, iter_num = 0, 0
    y_pred = th.zeros_like(labels).cpu()
    kld_weight = 0.1
    for batch in train_loader:
        # pdb.set_trace()
        sg = dgl.out_subgraph(graph, batch.to(device), relabel_nodes=True)
        # pred, mu, log_var = model(sg, feat[batch])
        pred, mu, log_var = model(sg, feat[sg.ndata['_ID']])
        pred = pred[:batch.shape[0]]
        mu = mu[:batch.shape[0]]
        log_var = log_var[:batch.shape[0]]
        y_pred[batch] = pred.argmax(dim=-1, keepdim=True).cpu()
        # y_pred.append(pred.argmax(dim=-1, keepdim=True).cpu())
        kld_loss = th.mean(-0.5 * th.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss_kl = kld_weight * kld_loss
        if focal_loss:
            loss = focal_cross_entropy(pred[:batch.shape[0]], labels[batch]) + loss_kl
        else : 
            loss = cross_entropy(pred[:batch.shape[0]], labels[batch]) + loss_kl
        print(loss)
        pdb.set_trace()
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter_num += 1
    loss_a = total_loss / iter_num
    return loss_a, y_pred


@th.no_grad()
def evaluate(model, graph, labels, loader, train_idx, val_idx, test_idx, use_labels, evaluator):
    model.eval()

    feat = graph.ndata["feat"]

    if use_labels:
        feat = add_labels(feat, labels, train_idx)
    # full-graph
    # pred, mu, log_var = model(graph, feat)
    # sub-graph
    pred = th.zeros(size=(labels.shape[0],n_classes), dtype=th.float).to(device)
    for batch in loader:
        sg = dgl.out_subgraph(graph, batch.to(device), relabel_nodes=True)
        p, mu, log_var = model(sg, feat[sg.ndata['_ID']])
        p = p[:batch.shape[0]]
        mu = mu[:batch.shape[0]]
        log_var = log_var[:batch.shape[0]]
        # pred[batch] = p.argmax(dim=-1, keepdim=True).cpu()
        pred[batch] = p

    train_loss = cross_entropy(pred[train_idx], labels[train_idx])
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])
    test_loss = cross_entropy(pred[test_idx], labels[test_idx])

    pred = pred.argmax(dim=-1, keepdim=True)

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
        pred
    )


def run(args, graph, labels, train_idx, val_idx, test_idx, train_loader, val_loader, test_loader, all_loader, evaluator, n_running):
    # define model and optimizer
    model = gen_model(args) 
    print(count_parameters(args))
    model = model.to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training loop
    total_time = 0
    best_val_acc, best_test_acc, best_val_loss = 0, 0, float("inf")
    best_out = None

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, args.n_epochs + 1):
        th.cuda.empty_cache()

        tic = time.time()

        adjust_learning_rate(optimizer, args.lr, epoch)

        loss, pred = train(model, graph, labels, train_idx, train_loader, optimizer, args.use_labels, args.focal_loss)
        # pdb.set_trace()
        tt = train_idx.cpu()
        acc = compute_acc(pred[tt], labels[tt], evaluator)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, out = evaluate(
            model, graph, labels, all_loader, train_idx, val_idx, test_idx, args.use_labels, evaluator
        )

        toc = time.time()
        total_time += toc - tic

        if test_acc > best_test_acc:
        # if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_out = out

        if epoch % args.log_every == 0:
            print(f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}")
            print(
                f"Loss: {loss.item():.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Best test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{best_test_acc:.4f}"
            )

        for l, e in zip(
            [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
            [acc, train_acc, val_acc, test_acc, loss.item(), train_loss, val_loss, test_loss],
        ):
            l.append(e)

    print("*" * 50)
    print(f"Average epoch time: {total_time / args.n_epochs}, Test acc: {best_test_acc}")

    if args.plot_curves:
        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.set_yticks(np.linspace(0, 1.0, 101))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip([accs, train_accs, val_accs, test_accs], ["acc", "train acc", "val acc", "test acc"]):
            plt.plot(range(args.n_epochs), y, label=label)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gat_acc_{n_running}.png")

        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip(
            [losses, train_losses, val_losses, test_losses], ["loss", "train loss", "val loss", "test loss"]
        ):
            plt.plot(range(args.n_epochs), y, label=label)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gat_loss_{n_running}.png")

    return best_val_acc, best_test_acc, best_out


def count_parameters(args):
    model = gen_model(args)
    print([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def main():
    global device, in_feats, n_classes, epsilon

    argparser = argparse.ArgumentParser("VAE on OGB", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--n-runs", type=int, default=4)
    argparser.add_argument("--n-epochs", type=int, default=2000)
    argparser.add_argument("--use-labels", action="store_true", help="Use labels in the training set as input features.")
    argparser.add_argument("--use-norm", action="store_true", help="Use symmetrically normalized adjacency matrix.")
    argparser.add_argument("--lr", type=float, default=0.002)
    argparser.add_argument("--n-layers", type=int, default=4)
    argparser.add_argument("--n-heads", type=int, default=3)
    argparser.add_argument("--n-hidden", type=int, default=512)
    argparser.add_argument("--dropout", type=float, default=0.75)
    argparser.add_argument("--attn_drop", type=float, default=0.05)
    argparser.add_argument("--wd", type=float, default=0)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--plot-curves", action="store_true")
    argparser.add_argument("--focal_loss", action="store_true")
    argparser.add_argument("--dataset", type=str, default="ogbn-products")
    argparser.add_argument("--batch_size", type=int, default=10000)
    args = argparser.parse_args()

    if args.cpu:
        device = th.device("cpu")
    else:
        device = th.device("cuda:%d" % args.gpu)

    # load data
    data = DglNodePropPredDataset(name=args.dataset)
    evaluator = Evaluator(name=args.dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    train_loader = th.utils.data.DataLoader(train_idx, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = th.utils.data.DataLoader(val_idx, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = th.utils.data.DataLoader(test_idx, batch_size=args.batch_size, shuffle=True, drop_last=False)
    all_loader = th.utils.data.DataLoader(th.arange(labels.shape[0]), batch_size=args.batch_size, shuffle=True, drop_last=False)

    # add reverse edges : arxiv
    srcs, dsts = graph.all_edges()
    if args.dataset == "ogbn-arxiv" :
        graph.add_edges(dsts, srcs)
    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    # srcs_ud, dsts_ud = graph.all_edges()
    # pkl_file = open(f'./dataset/ppmi_numwalks200_walklength5.pkl', 'rb')
    # ppmi_set = pkl.load(pkl_file)
    # ppmi = th.ones_like(srcs_ud)
    # pdb.set_trace()
    # for edge_index in tqdm(range(srcs_ud.shape[0])):
    #     if (srcs_ud[edge_index].item() , dsts_ud[edge_index].item()) in ppmi_set :
    #         ppmi[edge_index] = ppmi_set[(srcs_ud[edge_index].item() , dsts_ud[edge_index].item())]
    file = open(f'./dataset/ppmi_numwalks200_walklength5_{args.dataset}.pkl', 'rb')
    # pdb.set_trace()
    # pkl.dump(ppmi,file)
    ppmi = pkl.load(file)
    graph.edata.update({"ppmi" : ppmi})

    in_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()
    # graph.create_format_()
  
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)
    graph = graph.to(device)

    # run
    val_accs = []
    test_accs = []
    model_dir = f'../models/{args.dataset}_vae'
       
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    with open(f'{model_dir}/metadata', 'w') as f:
        f.write(f'# of params: {sum(p.numel() for p in gen_model(args).parameters())}\n')

    for i in range(1, args.n_runs + 1):
        val_acc, test_acc, out = run(args, graph, labels, train_idx, val_idx, test_idx, train_loader, val_loader, test_loader, all_loader, evaluator, i)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        th.save(F.softmax(out, dim=1), f'{model_dir}/{i-1}.pt')

    print(f"Runned {args.n_runs} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    print(f"Number of params: {count_parameters(args)}")

if __name__ == "__main__":
    main()
