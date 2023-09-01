

import argparse
import math
import time
import os
import shutil

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

def main():
    global device, in_feats_l, in_feats_r, n_classes_l, n_classes_r, epsilon

    argparser = argparse.ArgumentParser("Fuse 2 datasets of OGB", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--dataset_r", type=str, default="ogbn-products")
    argparser.add_argument("--dataset_l", type=str, default="ogbn-arxiv")
    argparser.add_argument("--fuse_dim", type=int, default=100)
    argparser.add_argument("--fuse_method", type=str, default="learn")
    argparser.add_argument("--root", type=str, default="./fuse/dataset/")
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

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    srcs_ud, dsts_ud = graph.all_edges()
    pkl_file = open(f'./dataset/ppmi_numwalks200_walklength5_{args.dataset}.pkl', 'rb')
    ppmi_set = pkl.load(pkl_file)
    ppmi = th.ones_like(srcs_ud)
    for edge_index in tqdm(range(srcs_ud.shape[0])):
        if (srcs_ud[edge_index].item() , dsts_ud[edge_index].item()) in ppmi_set :
            ppmi[edge_index] = ppmi_set[(srcs_ud[edge_index].item() , dsts_ud[edge_index].item())]
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

if __name__ == "__main__":
    main()
