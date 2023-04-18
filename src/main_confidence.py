import os
import sys
import yaml
import random
import resource
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


from args import parse_args
from filtering.dataset import get_confidence_loader
from model import load_model_for_training, to_cuda
from utils import printt, print_res, log, get_unixtime
from train_confidence import train, test_epoch
from helpers import WandbLogger, TensorboardLogger
from sample import sample
from data import load_data, get_data


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def main(args=None):
    print('Entering main')
    if args is None:
        args = parse_args()
    print(f'args.rmsd_classification_cutoff: {args.rmsd_classification_cutoff}')
    torch.cuda.set_device(args.gpu)
    torch.hub.set_dir(args.torchhub_path)

    # init wandb before too long data loading to avoid timeout error thrown by wandb
    if args.mode != "test" and args.num_folds==1:
        log_dir = os.path.join(args.tensorboard_path,
                                args.run_name,
                                get_unixtime())
        if args.logger == "tensorboard":
            writer = TensorboardLogger(log_dir=log_dir)
        elif args.logger == "wandb":
            writer = WandbLogger(project=args.project, 
                                entity=args.entity, 
                                name=args.run_name, 
                                group=args.group,
                                config=args)
        else:
            raise Exception("Improper logger.")

    # training mode, dump args for reproducibility
    if args.mode != "test":
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        # save args
        with open(args.args_file, "w+") as f:
            yaml.dump(args.__dict__, f)

    # load raw data
    if args.use_randomized_confidence_data:
        data = load_data(args)
        data_params = data.data_params
        loaders = get_data(data, 0, args)
        train_loader = loaders["train"]
        val_loader = loaders["val"]
    else:
        train_loader = get_confidence_loader("train", args, shuffle=True)
        val_loader = get_confidence_loader("val", args, shuffle=False)
        data_params = {'num_residues': 23}
    printt("finished loading raw data")

    # needs to be set if DataLoader does heavy lifting
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # needs to be set if sharing resources
    if args.num_workers >= 1:
        torch.multiprocessing.set_sharing_strategy("file_system")

    # train mode: train model for args.fold different seeds
    # and evaluate at the end
    if args.mode == "train":
        # save scores
        test_scores = defaultdict(list)
        # try different seeds
        for fold in range(args.num_folds):
            if args.num_folds > 1:
                log_dir = os.path.join(args.tensorboard_path,
                                        args.run_name, str(fold),
                                        get_unixtime())
                if args.logger == "tensorboard":
                    writer = TensorboardLogger(log_dir=log_dir)
                elif args.logger == "wandb":
                    writer = WandbLogger(project=args.project, 
                                        entity=args.entity, 
                                        name=args.run_name, 
                                        group=args.group,
                                        config=args)
                else:
                    raise Exception("Improper logger.")
            #### set up fold experiment
            set_seed(args.seed)
            # make save folder
            fold_dir = os.path.join(args.save_path, f"fold_{fold}")
            args.fold_dir = fold_dir
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)
            printt("fold {} seed {}\nsaved to {}".format(fold, args.seed, fold_dir))
            printt("finished creating data splits")
            # get model and load checkpoint, if relevant
            model = load_model_for_training(args, data_params, fold,confidence_mode=True)
            model = to_cuda(model, args)
            printt("finished loading model")

            numel = sum([p.numel() for p in model.parameters()])
            printt('Model with', numel, 'parameters')

            #### run training loop
            printt(f'len(train_loader): {len(train_loader)}')
            printt(f'len(val_loader): {len(val_loader)}')
            
            best_score, best_epoch, best_path = train(
                    train_loader, val_loader,
                    model, writer, fold_dir, args)
            printt("finished training best epoch {} loss {:.3f}".format(
                    best_epoch, best_score))

            #### run eval loop
            if best_path is not None:
                model = load_model_for_training(args, data_params, fold)
                model.load_state_dict(torch.load(best_path,
                    map_location="cpu")["model"])
                model = to_cuda(model, args)
                printt(f"loaded model from {best_path}")
            # eval on test set
            _,test_score = test_epoch(args, model, loaders["test"], writer)
            test_score["fold"] = fold
            # add val for hyperparameter search
            _,val_score = test_epoch(args, model, loaders["val"], writer)
            for key, val in val_score.items():
                test_score[f"val_{key}"] = val
            # print and save
            for key, val in test_score.items():
                test_scores[key].append(val)
            printt("fold {}".format(fold))
            print_res(test_score)
            # set next seed
            args.seed += 1
            break # run single fold
            # end of fold ========

        printt(f"{args.num_folds} folds average")
        print_res(test_scores)
        log(test_scores, args.log_file)
        # end of all folds ========

    # test mode: load up all replicates from checkpoint directory
    # and evaluate by sampling from reverse diffusion process
    elif args.mode == "test":
        set_seed(args.seed)
        printt("running inference")
        test_scores = defaultdict(list)
        for fold_dir in os.listdir(args.save_path):
            if "fold_" not in fold_dir:
                continue
            fold = int(fold_dir[5:])
            # load and convert data to DataLoaders
            
            printt("finished creating data splits")
            # get model and load checkpoint, if relevant
            model = load_model_for_training(args, data_params, fold,confidence_mode=True)
            model = to_cuda(model, args)
            printt("finished loading model")

            # run reverse diffusion process
            

            # test fold
            _,test_score = test_epoch(args, model, val_loader, writer=None) # TODO change to test_loader
            test_score["fold"] = fold
            # add val for hyperparameter search
            _,val_score = test_epoch(args, model, val_loader , writer=None)
            
            for key, val in val_score.items():
                test_score[f"val_{key}"] = val

            # print and save
            for key, val in test_score.items():
                test_scores[key].append(val)
            # end of fold ========

        printt(f"Average test/val performance")
        print_res(test_scores)
        log(test_scores, args.log_file, reduction=False)
        # end of all folds ========


    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # with open(f'{args.original_model_dir}/model_parameters.yml') as f:
    #     score_model_args = Namespace(**yaml.full_load(f))
    #     if not hasattr(score_model_args, 'separate_noise_schedule'):  # exists for compatibility
    #         score_model_args.separate_noise_schedule = False
    #     if not hasattr(score_model_args, 'lm_embeddings_path'):  # exists for compatibility
    #         score_model_args.lm_embeddings_path = None
    #     if not hasattr(score_model_args, 'all_atoms'):  # exists for compatibility
    #         score_model_args.all_atoms = False
    #     if not hasattr(score_model_args,'tr_only_confidence'):  # exists for compatibility
    #         score_model_args.tr_only_confidence = True
    #     if not hasattr(score_model_args,'high_confidence_threshold'):  # exists for compatibility
    #         score_model_args.high_confidence_threshold = 0.0
    #     if not hasattr(score_model_args, 'include_confidence_prediction'):  # exists for compatibility
    #         score_model_args.include_confidence_prediction = False
    #     if not hasattr(score_model_args, 'esm_embeddings_path'):  # exists for compatibility
    #         score_model_args.esm_embeddings_path = None

    # # construct loader
    # train_loader, val_loader = construct_loader_filtering(args, device)
    # model = get_model(score_model_args if args.transfer_weights else args, device, t_to_sigma=None, confidence_mode=True)
    # optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.main_metric_goal)

    # if args.transfer_weights:
    #     print("HAPPENING | Transferring weights from original_model_dir to the new model after using original_model_dir's arguments to construct the new model.")
    #     checkpoint = torch.load(os.path.join(args.original_model_dir,args.ckpt), map_location=device)
    #     model_state_dict = model.state_dict()
    #     transfer_weights_dict = {k: v for k, v in checkpoint.items() if k in list(model_state_dict.keys())}
    #     model_state_dict.update(transfer_weights_dict)  # update the layers with the pretrained weights
    #     model.load_state_dict(model_state_dict)

    # elif args.restart_dir:
    #     dict = torch.load(f'{args.restart_dir}/last_model.pt', map_location=torch.device('cpu'))
    #     model.module.load_state_dict(dict['model'], strict=True)
    #     optimizer.load_state_dict(dict['optimizer'])
    #     print("Restarting from epoch", dict['epoch'])

    # numel = sum([p.numel() for p in model.parameters()])
    # print('Model with', numel, 'parameters')
    # run_dir = os.path.join(args.log_dir, args.run_name)

    # if not args.no_train:
    #     if args.wandb:
    #         wandb.init(
    #             entity='coarse-graining-mit',
    #             settings=wandb.Settings(start_method="fork"),
    #             project=args.project,
    #             name=args.run_name,
    #             config=args
    #         )
    #         wandb.log({'numel': numel})

    #     # record parameters
    #     yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
    #     save_yaml_file(yaml_file_name, args.__dict__)
    #     args.device = device

    #     train(args, model, optimizer, scheduler, train_loader, val_loader, run_dir)

    #if args.test:
    #    test(args, model, val_loader, run_dir, multiplicity=args.multiplicity_test)

    
if __name__ == '__main__':
    main()