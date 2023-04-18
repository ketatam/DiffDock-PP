import gc
import os

import torch.nn.functional as F

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from train import save_model

torch.multiprocessing.set_sharing_strategy('file_system')

from utils import printt, get_optimizer


def train_epoch(args, model, loader, optimizer, writer, num_batches):
    model.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_labels = []
    all_pred = []
    all_loss = []
    for data in tqdm(loader, total=len(loader)):
        rmsd = torch.tensor([sample.rmsd for sample in data])
        # print('batch_0_0_ligand', batch[0][0]['ligand'].pos)
        # print('batch_0_0_receptor', batch[0][0]['receptor'].pos)
        # print('batch_0_1_ligand', batch[0][1]['ligand'].pos)
        # print('batch_0_1_receptor', batch[0][1]['receptor'].pos)
        # print('batch_0_0', batch[0][0])
        # print('batch_0_1', batch[0][1])

        # print('batch_0', batch[0])
        # print('batch_1', batch[1])
        # raise RuntimeError
        #data, rmsd = batch # TODO
        # move to CUDA
        if args.num_gpu == 1 and torch.cuda.is_available():
            data = data.cuda()

        optimizer.zero_grad()
        torch.cuda.empty_cache()
        try:
            pred = model(data)
            if args.rmsd_prediction:
                labels = rmsd.to(device)
                confidence_loss = F.mse_loss(pred, labels)
            else:
                #if isinstance(args.rmsd_classification_cutoff, list):
                #    labels = torch.cat([graph.y_binned for graph in data]).to(device)
                #    confidence_loss = F.cross_entropy(pred, labels)
                #else:
                labels = (rmsd < args.rmsd_classification_cutoff).float()
                if args.num_gpu == 1 and torch.cuda.is_available():
                    labels = labels.to(device)
                #print(f'labels: {labels}')
                #print(f'pred: {pred}')
                #print(f'rmsd: {rmsd}')
                confidence_loss = F.binary_cross_entropy_with_logits(pred, labels.to(pred.device))
                #accuracy = torch.mean((labels == (pred > 0).int()).float())
                #print(f'train_accuracy: {accuracy}')
            loss = confidence_loss
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            all_loss.append(loss.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_pred.append(pred.detach().cpu())

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e

        # write to tensorboard
        num_batches += 1
        loss = loss.cpu().item()  # CPU for logging
        if num_batches % args.log_frequency == 0:
            log_key = f"train_loss_per_batch"
            if writer is not None:
                writer.add_scalar(log_key, loss, num_batches)
        
    all_labels = torch.cat(all_labels)
    print(f'percentage of positives in train: {all_labels.mean()}')
    if writer:
        writer.add_scalar('train_gt_pos_perc', all_labels.mean())
    all_pred = torch.cat(all_pred)
    if not args.rmsd_prediction:
        accuracy = torch.mean((all_labels == (all_pred > 0).int()).float())
        print(f'train_accuracy: {accuracy}')
        if writer:
            writer.add_scalar('train_accuracy', accuracy)
    all_loss = torch.tensor(all_loss)
    print(f'train_loss_total: {all_loss.mean()}')
    if writer:
        writer.add_scalar('train_loss_total', all_loss.mean())

    return all_loss.mean()


def test_epoch(args, model, loader, writer):
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_labels = []
    all_pred = []
    all_loss = []
    for data in tqdm(loader, total=len(loader)):
        rmsd = torch.tensor([sample.rmsd for sample in data])
        #data, rmsd = batch
        # move to CUDA
        if args.num_gpu == 1 and torch.cuda.is_available():
            data = data.cuda()

        try:
            with torch.no_grad():
                pred = model(data)
            
            if args.rmsd_prediction:
                labels = rmsd.to(device)
                confidence_loss = F.mse_loss(pred, labels)
            else:
                #if isinstance(args.rmsd_classification_cutoff, list):
                #    labels = torch.cat([graph.y_binned for graph in data]).to(device)
                #    confidence_loss = F.cross_entropy(pred, labels)
                labels = (rmsd < args.rmsd_classification_cutoff).float()
                if args.num_gpu == 1 and torch.cuda.is_available():
                    labels = labels.to(device)
                #print(f'val_labels: {labels}')
                #print(f'val_pred: {pred}')
                confidence_loss = F.binary_cross_entropy_with_logits(pred, labels.to(pred.device))
                #try:
                #    roc_auc = roc_auc_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy())
                #except ValueError as e:
                #    if 'Only one class present in y_true. ROC AUC score is not defined in that case.' in str(e):
                #        roc_auc = 0
                #    else:
                #        raise e
            loss = confidence_loss

            all_labels.append(labels.detach().cpu())
            all_pred.append(pred.detach().cpu())
            all_loss.append(loss.detach().cpu().item())

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    all_labels = torch.cat(all_labels)
    print(f'percentage of positives in val: {all_labels.mean()}')
    if writer:
        writer.add_scalar('val_gt_pos_perc', all_labels.mean())
    all_pred = torch.cat(all_pred)
    all_loss = torch.tensor(all_loss)
    accuracy = None
    if args.rmsd_prediction:
        baseline_metric = ((all_labels - all_labels.mean()).abs()).mean()
    else:
        baseline_metric = all_labels.sum() / len(all_labels)
        accuracy = torch.mean((all_labels == (all_pred > 0).int()).float())
        print(f'val_accuracy: {accuracy}')
        if writer:
            writer.add_scalar('val_accuracy', accuracy)

    average_loss = all_loss.mean()
    print(f'val_loss: {average_loss}')
    if writer:
        writer.add_scalar('val_loss', average_loss)

    return average_loss, accuracy


def train(train_loader, val_loader, model, writer, fold_dir, args):
    # optimizer
    start_epoch, optimizer = get_optimizer(model, args, load_best=True, confidence_mode=True)

    # validation
    best_loss = float("inf")
    best_metrics = {'accuracy': 0.0}

    best_path = None
    best_epoch = start_epoch

    num_batches = start_epoch * (len(train_loader) // args.batch_size)
    ep_iterator = range(start_epoch, start_epoch+args.epochs)
    if not args.no_tqdm:
        ep_iterator = tqdm(ep_iterator,
                           initial=start_epoch,
                           desc="train epoch", ncols=50)

    # test once before starting training
    val_loss, val_accuracy = test_epoch(args, model, val_loader, writer)
    for epoch in ep_iterator:
        writer.add_scalar("epoch", epoch)
        # start epoch!
        train_loss = train_epoch(args, model, train_loader, optimizer, writer, num_batches)

        # evaluate (end of epoch)
        val_loss, val_accuracy = test_epoch(args, model, val_loader, writer)

        printt('\nepoch', epoch)
        printt('train loss', train_loss)
        printt('val loss', val_loss)
        print("val_accuracy", val_accuracy)

        # save latest model every e.g. 10 epochs
        if epoch % args.save_model_every == 0 and epoch != 0:
            last_path = os.path.join(fold_dir, "model_last.pth")
            save_model(model, args, optimizer, last_path)


        # save model if it improves rmsd
        if val_loss < best_loss:
            best_loss = val_loss
            best_metrics['accuracy'] = val_accuracy

            best_epoch = epoch

            path_suffix = f"{num_batches}_{epoch}_{best_loss:.3f}_{best_metrics['accuracy']:.3f}.pth"
            # save model ONLY IF best
            best_path = os.path.join(fold_dir, f"model_best_{path_suffix}")
            save_model(model, args, optimizer, best_path)

        # check if out of patience
        if epoch - best_epoch >= args.patience:
            break


        # end of epoch ========
    # end of all epochs ========

    return best_loss, best_epoch, best_path
