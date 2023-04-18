import os
import sys
import glob
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.data_parallel import DataParallel

from utils import printt, get_model_path
from .model import ScoreModel,ConfidenceModel
from notebooks.utils_notebooks import Dict2Class


def load_model(args, model_params, fold, load_best=True, confidence_mode = False):
    """
        Model factory
        :load_best: if True, load best model in terms of performance on val set, else load last model
    """
    # load model args
    
    with open(os.path.join(args.filtering_model_path if confidence_mode else args.score_model_path,"../args.yaml")) as f:
        model_args = yaml.safe_load(f)
    model_args = Dict2Class(model_args)
    model_args.gpu = args.gpu
    model_args.num_gpu = args.num_gpu


    # load model with specified arguments
    kwargs = {}
    if args.model_type == "e3nn":
        if confidence_mode:
            model = ConfidenceModel(model_args, model_params, **kwargs)
        else:
            model = ScoreModel(model_args, model_params, **kwargs)
    else:
        raise Exception(f"invalid model type {args.model_type}")
    printt("loaded model with kwargs:", " ".join(kwargs.keys()))

    checkpoint=None
    # (optional) load checkpoint if provided
    if confidence_mode and args.filtering_model_path is not None:
        load_best = True # TODO: Remove
        if load_best:
            checkpoint = select_model(args.filtering_model_path,True)
        else:
            checkpoint = os.path.join(args.filtering_model_path, "model_last.pth")
    elif not confidence_mode and args.score_model_path is not None:
        if load_best:
            checkpoint = select_model(args.score_model_path,False)
        else:
            checkpoint = os.path.join(args.score_model_path, "model_last.pth")
    elif args.checkpoint_path is not None:
        fold_dir = os.path.join(args.checkpoint_path, f"fold_{fold}")
        if load_best:
            checkpoint = get_model_path(fold_dir)
        else:
            checkpoint = os.path.join(fold_dir, "model_last.pth")
    
    print("checkpoint",checkpoint)
      
    if checkpoint is not None:
        # extract current model
        state_dict = model.state_dict()
        # load onto CPU, transfer to proper GPU
        pretrain_dict = torch.load(checkpoint, map_location="cpu")["model"]
        pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in state_dict}
        # update current model
        state_dict.update(pretrain_dict)
        # >>>
        for k,v in state_dict.items():
            if k not in pretrain_dict:
                print(k, "not saved")
        model.load_state_dict(state_dict)
        printt("loaded checkpoint from", checkpoint)
    else:
        printt("no checkpoint found")

    return model

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

def load_model_for_training(args, model_params, fold, load_best=True, confidence_mode = False):
    """
        Model factory
        :load_best: if True, load best model in terms of performance on val set, else load last model
    """
    # load model with specified arguments
    kwargs = {}
    if args.model_type == "e3nn":
        if confidence_mode:
            model = ConfidenceModel(args, model_params, **kwargs)
        else:
            model = ScoreModel(args, model_params, **kwargs)
    else:
        raise Exception(f"invalid model type {args.model_type}")
    printt("loaded model with kwargs:", " ".join(kwargs.keys()))

    # (optional) load checkpoint if provided
    if args.checkpoint_path is not None:
        fold_dir = os.path.join(args.checkpoint_path, f"fold_{fold}")
        if load_best:
            checkpoint = select_model(fold_dir, confidence_mode=confidence_mode)
        else:
            checkpoint = os.path.join(fold_dir, "model_last.pth")
            
        if checkpoint is not None:
            # extract current model
            state_dict = model.state_dict()
            # load onto CPU, transfer to proper GPU
            pretrain_dict = torch.load(checkpoint, map_location="cpu")["model"]
            pretrain_dict = {k:v for k,v in pretrain_dict.items() if k in state_dict}
            # update current model
            state_dict.update(pretrain_dict)
            # >>>
            for k,v in state_dict.items():
                if k not in pretrain_dict:
                    print(k, "not saved")
            model.load_state_dict(state_dict)
            printt("loaded checkpoint from", checkpoint)
        else:
            printt("no checkpoint found")
    return model


def to_cuda(model, args):
    """
        move model to cuda
    """
    # specify number in case test =/= train GPU
    if args.gpu >= 0:
        model = model.cuda(args.gpu)
        if args.num_gpu > 1:
            device_ids = [args.gpu + i for i in range(args.num_gpu)]
            model = DataParallel(model, device_ids=device_ids)
    return model


