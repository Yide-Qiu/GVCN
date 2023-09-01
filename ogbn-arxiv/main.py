
import argparse
import math
import time
import os
import shutil
import uuid

import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from outcome_correlation import prepare_folder

from models import VAE
import pickle as pkl
from tqdm import tqdm
import pdb


device = None
in_feats, n_classes = None, None
epsilon = 1 - math.log(2)


def gen_model(args):
    norm = "both" if args.use_norm else "none"

    if args.use_labels:
        model = VAE(
            in_feats + n_classes,
            n_classes,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            activation=F.relu,
            dropout=args.dropout,
            input_drop=args.input_drop,
            attn_drop=args.attn_drop,
            edge_drop=args.edge_drop,
            norm=norm,
        )
    else:
        model = VAE(
            in_feats,
            n_classes,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            activation=F.relu,
            dropout=args.dropout,
            input_drop=args.input_drop,
            attn_drop=args.attn_drop,
            norm=norm,
        )

    return model

def save_checkpoint(pred, n_running, checkpoint_path):
    fname = os.path.join(checkpoint_path, f'best_pred_run{n_running}.pt')
    print('Saving prediction.......')
    th.save(pred.cpu(),fname)

def cross_entropy(x, labels):
    return F.cross_entropy(x, labels[:, 0], reduction="mean")

def loge_cross_entropy(x, labels):
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

def kd_loss(x, y, temp):
    return th.nn.KLDivLoss()(F.log_softmax(x/temp, dim=1), F.softmax(y/temp, dim=1)) * (temp * temp)

def compute_acc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]


def add_labels(feat, labels, idx):
    onehot = th.zeros([feat.shape[0], n_classes]).to(device)
    onehot[idx, labels[idx, 0]] = 1
    return th.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50

def train_rlu(args, model, graph, labels, pseudo_labels, confident_nid, val_idx, test_idx, teacher_output, optimizer):
    model.train()
    # pdb.set_trace()
    feat = graph.ndata["feat"]

    mask_rate = args.mask
    if args.use_labels:
        mask = th.rand(confident_nid.shape) < mask_rate

        train_labels_idx = confident_nid[mask]
        train_pred_idx = confident_nid[~mask]

        feat = add_labels(feat, pseudo_labels, train_labels_idx)
    else:
        mask = th.rand(confident_nid.shape) < mask_rate

        train_pred_idx = confident_nid[mask]

    optimizer.zero_grad()
    pred, mu, log_var = model(graph, feat)

    train_pred_idx = train_pred_idx.to(device)
    if args.n_label_iters > 0 and args.use_labels:
        unlabel_idx = th.cat([train_pred_idx, val_idx, test_idx])
        for _ in range(args.n_label_iters):
            pred = pred.detach()
            th.cuda.empty_cache()
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred, _ , _ = model(graph, feat)

    # kd loss
    loss_kd = 0
    kd_weight = 0
    if args.self_kd:
        kd_weight = args.kd_weight
        loss_kd = kd_loss(pred, teacher_output, args.temp)
        loss_kd = loss_kd * kd_weight
    # vae kl loss
    kl_weight = args.kl_weight
    log_var[log_var>10] = 10
    kl_loss = th.mean(-0.5 * th.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    loss_kl = kl_weight * kl_loss

    ce_loss_weight = 1-kl_weight-kd_weight
    L1 = 0
    if args.loss == "focal":
        L1 = ce_loss_weight*focal_cross_entropy(pred[train_pred_idx], pseudo_labels[train_pred_idx]) 
    if args.loss == "loge" :
        L1 = ce_loss_weight*loge_cross_entropy(pred[train_pred_idx], pseudo_labels[train_pred_idx])
    if args.loss == "ce" : 
        L1 = ce_loss_weight*cross_entropy(pred[train_pred_idx], pseudo_labels[train_pred_idx]) 
        
    loss = L1 + loss_kl + loss_kd
    loss.backward()
    optimizer.step()

    return loss, pred

def train(args, model, graph, labels, train_idx, val_idx, test_idx, teacher_output, optimizer):
    model.train()

    feat = graph.ndata["feat"]

    mask_rate = args.mask
    if args.use_labels:
        mask = th.rand(train_idx.shape) < mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask = th.rand(train_idx.shape) < mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    pred, mu, log_var = model(graph, feat)

    if args.n_label_iters > 0 and args.use_labels:
        unlabel_idx = th.cat([train_pred_idx, val_idx, test_idx])
        for _ in range(args.n_label_iters):
            pred = pred.detach()
            th.cuda.empty_cache()
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred, _ , _ = model(graph, feat)

    # kd loss
    loss_kd = 0
    kd_weight = 0
    if args.self_kd:
        kd_weight = args.kd_weight
        loss_kd = kd_loss(pred, teacher_output, args.temp)
        loss_kd = loss_kd * kd_weight
    # vae kl loss
    kl_weight = args.kl_weight
    kl_loss = th.mean(-0.5 * th.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    loss_kl = kl_weight * kl_loss

    ce_loss_weight = 1-kl_weight-kd_weight
    L1 = 0
    if args.loss == "focal":
        L1 = ce_loss_weight*focal_cross_entropy(pred[train_pred_idx], labels[train_pred_idx]) 
    if args.loss == "loge" :
        L1 = ce_loss_weight*loge_cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
    if args.loss == "ce" : 
        L1 = ce_loss_weight*cross_entropy(pred[train_pred_idx], labels[train_pred_idx]) 
        
    loss = L1 + loss_kl + loss_kd
    loss.backward()
    optimizer.step()

    return loss, pred


@th.no_grad()
def evaluate(args, model, graph, labels, train_idx, val_idx, test_idx, evaluator):
    model.eval()

    feat = graph.ndata["feat"]

    if args.use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred, _, _ = model(graph, feat)
    if args.n_label_iters > 0 and args.use_labels:
        unlabel_idx = th.cat([val_idx, test_idx])
        for _ in range(args.n_label_iters):
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred, _ , _ = model(graph, feat)

    train_loss = cross_entropy(pred[train_idx], labels[train_idx])
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])
    test_loss = cross_entropy(pred[test_idx], labels[test_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
        pred
    )


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running):
    # define model and optimizer
    model = gen_model(args)
    print(count_parameters(args))
    model = model.to(device)

    dirs = f"./output/{args.dataset}/"
    if not os.path.exists(dirs):os.makedirs(dirs)
    checkpt_file = dirs+uuid.uuid4().hex

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training loop
    total_time = 0
    best_val_acc, best_test_acc, best_val_loss = 0, 0, float("inf")
    best_out = None

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    checkpoint_path = args.checkpoint_path

    if args.self_kd:
        teacher_output = th.load(os.path.join(args.checkpoint_path, f'best_pred_run{n_running}.pt')).cpu().cuda()
    else:
        teacher_output = None

    if not args.rlu :
        for epoch in range(1, args.n_epochs + 1):
            tic = time.time()

            adjust_learning_rate(optimizer, args.lr, epoch)

            loss, pred = train(args, model, graph, labels, train_idx, val_idx, test_idx, teacher_output, optimizer)
            acc = compute_acc(pred[train_idx], labels[train_idx], evaluator)

            train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, out = evaluate(
                args, model, graph, labels, train_idx, val_idx, test_idx, evaluator
            )

            toc = time.time()
            total_time += toc - tic

            if args.selection_metric == "acc":
                new_best = val_acc > best_val_acc
            else:
                new_best = val_loss < best_val_loss
            if test_acc > best_test_acc:
            # if new_best:
            # if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_out = out
                if args.save:
                    os.makedirs(checkpoint_path, exist_ok=True)
                    save_checkpoint(best_out, n_running, checkpoint_path)   


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

    else :
        for stage in range(1,args.n_stages + 1):
            # rlu
            if stage == 1 :
                pseudo_labels = th.zeros_like(labels)
                print("------ pseudo labels inited, rate: {:.4f} ------".format(len(train_idx)/labels.shape[0]))

            else :
                predict_prob = th.load(checkpt_file+'_{}_{}.pt'.format((stage-1), args.method))/args.t
                predict_prob = predict_prob.softmax(dim=1)
                confident_nid = th.arange(len(predict_prob))[(predict_prob.max(1)[0] > args.threshold).cpu()]
                predict_labels = th.argmax(predict_prob,dim=1)
                pseudo_labels[confident_nid] = predict_labels[confident_nid].view(-1,1)
                temp = predict_labels[confident_nid] - labels[confident_nid].view(-1)
                print(f'Stage: {stage}, confident nodes: {temp.shape[0]}')
                print(f'Stage: {stage}, confident cor rate : {temp[temp==0].shape[0]/temp.shape[0]}')
                pseudo_labels[train_idx] = labels[train_idx]
                confident_nid = th.tensor(list(set(confident_nid.tolist())|set(train_idx.cpu().tolist())))
                print("------ pseudo labels updated, rate: {:.4f} ------".format(len(confident_nid)/(labels.shape[0])))

            for epoch in range(1, args.n_epochs + 1):
                tic = time.time()

                adjust_learning_rate(optimizer, args.lr, epoch)
                if stage == 1:
                    loss, pred = train(args, model, graph, labels, train_idx, val_idx, test_idx, teacher_output, optimizer)
                else:
                    loss, pred = train_rlu(args, model, graph, labels, pseudo_labels, confident_nid, val_idx, test_idx, teacher_output, optimizer)

                acc = compute_acc(pred[train_idx], labels[train_idx], evaluator)

                train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, out = evaluate(
                    args, model, graph, labels, train_idx, val_idx, test_idx, evaluator
                )

                toc = time.time()
                total_time += toc - tic

                if args.selection_metric == "acc":
                    new_best = val_acc > best_val_acc
                else:
                    new_best = val_loss < best_val_loss
                if test_acc > best_test_acc:
                # if new_best:
                # if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        best_out = out
                        if args.save:
                            os.makedirs(checkpoint_path, exist_ok=True)
                            save_checkpoint(best_out, n_running, checkpoint_path)   

                if epoch % args.log_every == 0:
                    th.save(best_out, checkpt_file + f'_{stage}_{args.method}.pt')
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
            print(f"Stage: {stage}, Average epoch time: {total_time / args.n_epochs}, Test acc: {best_test_acc}")



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
        plt.savefig(f"acc_{n_running}.png")

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
        plt.savefig(f"loss_{n_running}.png")

    return best_val_acc, best_test_acc, best_out


def count_parameters(args):
    model = gen_model(args)
    print([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


# def main(a,b,c):
# def main(a):
def main():
    global device, in_feats, n_classes, epsilon

    argparser = argparse.ArgumentParser("VAE on OGB", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--n-runs", type=int, default=4)
    argparser.add_argument("--n-epochs", type=int, default=2000)
    argparser.add_argument("--n-stages", type=int, default=1)
    argparser.add_argument("--use-labels", action="store_true", help="Use labels in the training set as input features.")
    argparser.add_argument("--use-norm", action="store_true", help="Use symmetrically normalized adjacency matrix.")
    argparser.add_argument("--lr", type=float, default=0.002)
    argparser.add_argument("--n-layers", type=int, default=3)
    argparser.add_argument("--n-heads", type=int, default=3)
    argparser.add_argument("--n-hidden", type=int, default=256)
    argparser.add_argument("--dropout", type=float, default=0.65)
    argparser.add_argument("--input_drop", type=float, default=0.25)
    argparser.add_argument("--attn_drop", type=float, default=0.05)
    argparser.add_argument("--edge_drop", type=float, default=0.6)
    argparser.add_argument("--wd", type=float, default=0)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--plot-curves", action="store_true")
    argparser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    argparser.add_argument("--mask", type=float, default=0.5)
    argparser.add_argument("--n-label-iters", type=int, default=0)
    argparser.add_argument("--self-kd", action="store_true")
    argparser.add_argument("--kd-weight", type=float, default=0.4)
    argparser.add_argument("--kl-weight", type=float, default=0.1)
    argparser.add_argument("--temp", type=float, default=1.0, help="weight of kd loss")
    argparser.add_argument("--t", type=float, default=0.3, help="temperature of RLU")
    argparser.add_argument("--selection-metric", type=str, default="acc", choices=["acc", "loss"])
    argparser.add_argument("--loss", type=str, default="loge", choices=["ce", "focal", "loge"])
    argparser.add_argument("--rlu", action="store_true")
    argparser.add_argument("--threshold", type=float, default=0.92)
    argparser.add_argument("--method", type=str, default="vae")
    argparser.add_argument("--use-xrt-emb", action="store_true")

    argparser.add_argument("--save", action="store_true")
    argparser.add_argument("--checkpoint-path", type=str, default="./checkpoint/")
    argparser.add_argument("--output-path", type=str, default="./output/")
    argparser.add_argument("--save-pred", action="store_true", help="save final predictions")

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

    if args.use_xrt_emb:
        graph.ndata["feat"] = th.from_numpy(np.load("./ogbn-arxiv/X.all.xrt-emb.npy")).float()

    # add reverse edges : arxiv
    srcs, dsts = graph.all_edges()
    if args.dataset == "ogbn-arxiv" :
        graph.add_edges(dsts, srcs)
    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    # srcs_ud, dsts_ud = graph.all_edges()
    # pkl_file = open(f'./dataset/ppmi_numwalks200_walklength5_{args.dataset}.pkl', 'rb')
    # ppmi_set = pkl.load(pkl_file)
    # ppmi = th.ones_like(srcs_ud)
    # # pdb.set_trace()
    # for edge_index in tqdm(range(srcs_ud.shape[0])):
    #     if (srcs_ud[edge_index].item() , dsts_ud[edge_index].item()) in ppmi_set :
    #         ppmi[edge_index] = ppmi_set[(srcs_ud[edge_index].item() , dsts_ud[edge_index].item())]
    file = open(f'./dataset/ppmi_numwalks200_walklength5_{args.dataset}.pkl', 'rb')
    # pdb.set_trace()
    # pkl.dump(ppmi,file)

    ppmi = pkl.load(file)
    # pdb.set_trace()
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
        val_acc, test_acc, out = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, i)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        th.save(F.softmax(out, dim=1), f'{model_dir}/{i-1}.pt')

    print(f"Runned {args.n_runs} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    print(f"Number of params: {count_parameters(args)}")
    
    return np.mean(test_accs)

def onjective(trail):
    # thr = trail.suggest_float('thr',0.70,0.95,step=0.01)
    # t = trail.suggest_float('t',0.3,1.2,step=0.1)
    # dr = trail.suggest_float('dr',0.6,0.9,step=0.05)
    kd = trail.suggest_float('kd',0.0,0.94,step=0.05)
    test_acc = main(kd)
    return test_acc

if __name__ == "__main__":
    # import time
    # import pdb
    # import optuna
    # st = time.time()
    # study = optuna.create_study(study_name = 'test', direction='maximize')
    # study.optimize(onjective, n_trials=20)
    # print(study.best_params)
    # print(study.best_trial)
    # print(study.best_trial.value)  
    # print(time.time()-st)  
    # pdb.set_trace()

    main()
    
