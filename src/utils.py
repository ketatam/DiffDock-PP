import os
import csv
import yaml
import glob
import itertools
from collections import defaultdict

import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

# -------- general

def load_csv(fp, split=None, batch=None):
    data = []
    with open(fp) as f:
        reader = csv.DictReader(f)
        for line in reader:
            # Skip not specified batches & splits
            if split is not None and line["split"] != split:
                continue
            if batch is not None and int(line["batch"]) != int(batch):
                continue
            data.append(line)
    return data


def log(item, fp, reduction=True):
    # pre-process item
    item_new = {}
    for key, val in item.items():
        if type(val) is list and reduction:
            key_std = f"{key}_std"
            item_new[key] = float(np.mean(val))
            item_new[key_std] = float(np.std(val))
        else:
            if torch.is_tensor(val):
                item[key] = val.tolist()
            item_new[key] = val
    # initialization: write keys
    if not os.path.exists(fp):
        with open(fp, "w+") as f:
            f.write("")
    # append values
    with open(fp, "a") as f:
        yaml.dump(item_new, f)
        f.write(os.linesep)


def chain(iterable, as_set=True):
    if as_set:
        return sorted(set(itertools.chain.from_iterable(iterable)))
    else:
        return list(itertools.chain.from_iterable(iterable))


def get_timestamp():
    return datetime.now().strftime('%H:%M:%S')


def get_unixtime():
    timestamp = str(int(time.time()))
    return timestamp


def printt(*args, **kwargs):
    print(get_timestamp(), *args, **kwargs)


def print_res(scores):
    """
        @param (dict) scores key -> score(s)
    """
    for key, val in scores.items():
        if type(val) is list:
            print_str = f"{np.mean(val):.3f} +/- {np.std(val):.3f}"
            print_str = print_str + f" ({len(val)})"
        else:
            print_str = f"{val:.3f}"
        print(f"{key}\t{print_str}")


def get_model_path(fold_dir):
    # load last model saved (we only save if improvement in validation performance)
    # convoluted code says "sort by epoch, then batch"
    # new code says "sort by rmsd, take the lowest"
    paths = []
    for path in glob.glob(f"{fold_dir}/*.pth"):
        if "last" not in path:
            paths.append(path)
    models = sorted(paths, key=lambda s:float(s.split("/")[-1].split("_")[4]))
                    #key=lambda s:(int(s.split("/")[-1].split("_")[3]),
                    #              int(s.split("/")[-1].split("_")[2])))
    if len(models) == 0:
        print(f"no models found at {fold_dir}")
        return
    checkpoint = models[0]
    return checkpoint

def select_model(fold_dir,confidence_mode):
    paths = []
    for path in glob.glob(f"{fold_dir}/*.pth"):
        if "last" not in path:
            paths.append(path)

    if confidence_mode:
        models = sorted(paths, key=lambda s:-float(s.split("/")[-1].split("_")[-1][:-4]))
    else:
        models = sorted(paths, key=lambda s:float(s.split("/")[-1].split("_")[4]))

    if len(models) == 0:
        print(f"no models found at {fold_dir}")
        return
    checkpoint = models[0]
    return checkpoint



def get_optimizer(model, args, load_best=True, confidence_mode=False):
    """
        Initialize optimizer and load if applicable
    """
    optimizer = Adam(model.parameters(),
                     lr=args.lr,
                     weight_decay=args.weight_decay)
    #optimizer = SGD(model.parameters(),
    #                lr=args.lr,
    #                momentum=0.5,
    #                weight_decay=args.weight_decay)
    # SGD is awful for my models. don't use it.
    ## load optimizer state
    fold_dir = args.fold_dir
    if args.checkpoint_path is not None:
        if load_best:
            checkpoint = select_model(fold_dir, confidence_mode=confidence_mode)
        else:
            checkpoint = os.path.join(fold_dir, "model_last.pth")

        if checkpoint is not None:
            #start_epoch = int(checkpoint.split("/")[-1].split("_")[3])
            start_epoch = 0
            with torch.no_grad():
                optimizer.load_state_dict(torch.load(checkpoint,
                    map_location="cpu")["optimizer"])
            printt("Finished loading optimizer")
        else:
            start_epoch = 0
    else:
        start_epoch = 0
    return start_epoch, optimizer


def init(model):
    """
        Wrapper around Xavier normal initialization
        Apparently this needs to be called in __init__ lol
    """
    for name, param in model.named_parameters():
        # NOTE must name parameter "bert"
        if "bert" in name:
            continue
        # bias terms
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        # weight terms
        else:
            nn.init.xavier_normal_(param)

# -------- metrics

def compute_rmsd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dist = ((x-y)**2).sum(-1)
    dist = dist / len(dist)  # normalize
    dist = dist.sum().sqrt()
    return dist


def compute_metrics(true, pred):
    """
        this function needs to be overhauled

        these lists are JAGGED IFF as_sequence=True
        @param pred (n, sequence, 1) preds for prob(in) where in = 1
        @param true (n, sequence, 1) targets, binary vector
    """
    # metrics depend on task
    as_sequence = type(true[0]) is list
    if as_sequence:
        f_metrics = {
            "roc_auc": _compute_roc_auc,
            "prc_auc": _compute_prc_auc
        }
    else:
        as_classification = (type(true[0]) == torch.Tensor
                             and true[0].dtype == torch.long)
        if as_classification:
            f_metrics = {
                "topk_accuracy": _compute_topk
            }
        else:
            f_metrics = {
                "mse": _compute_mse,
            }
    scores = defaultdict(list)
    for key, f in f_metrics.items():
        if as_sequence:
            for t,p in zip(true, pred):
                scores[key].append(f(t, p))
            scores[key] = np.mean(scores[key])
        else:
            if as_classification:
                topk = f(true, pred)
                ks = [1, 5, 10]
                for i,val in enumerate(topk):
                    scores[f"{key}_{ks[i]}"] = val
            else:
                scores[key] = f(true, pred)
    return scores


def _compute_roc_auc(true, pred):
    try:
        return metrics.roc_auc_score(true, pred)
    except:
        # single target value
        return 0.5


def _compute_prc_auc(true, pred):
    if true.sum() == 0:
        return 0.5
    precision, recall, _ = metrics.precision_recall_curve(true, pred)
    prc_auc = metrics.auc(recall, precision)
    return prc_auc


def _compute_mse(true, pred):
    # technically order doesn't matter but "input" then "target"
    true, pred = torch.tensor(true), torch.tensor(pred)
    return F.mse_loss(pred, true).item()


def _compute_topk(true, pred, topk=[1, 5, 10]):
    """
        @param (list)  topk
    """
    if type(true) is list:
        true, pred = torch.stack(true), torch.stack(pred)
    true, pred = true.cpu().numpy(), pred.cpu().numpy()
    labels = np.arange(pred.shape[-1])
    topk_accs = []
    for k in topk:
        # NOTE this does not handle duplicates.
        # "correct" predictions are sorted by index
        acc = metrics.top_k_accuracy_score(true, pred, k=k, labels=labels)
        topk_accs.append(acc)
    return topk_accs


if __name__ == "__main__":

    # test topk
    true = torch.arange(5)
    pred = torch.eye(5)
    topk = _compute_topk(true, pred, topk=[1,3,5])
    print(topk)

    true = torch.arange(4) + 1
    pred = torch.eye(5)[:4]
    topk = _compute_topk(true, pred, topk=[1,3,4])
    print(topk)

