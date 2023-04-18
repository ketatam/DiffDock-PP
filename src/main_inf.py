import os
import sys
import yaml
import random
import resource
from collections import defaultdict
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import wandb
import traceback
import time

from args import parse_args
from data import load_data, get_data,BindingDataset
from model import load_model, to_cuda
from utils import printt, print_res, log, get_unixtime
from train import train, evaluate, evaluate_pose
from filtering.dataset import get_confidence_loader
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
from geom_utils import set_time
# from helpers import WandbLogger, TensorboardLogger
from sample import sample
from evaluation.compute_rmsd import evaluate_all_rmsds


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def evaluate_confidence(model,loader,args):
    
    all_confidences = []
    all_confidences_with_name ={}
    

    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_labels = []
    all_pred = []
    all_loss = []
    print("loader len: ",len(loader))
    for data in tqdm(loader, total=len(loader)):
        #data, rmsd = batch
        # move to CUDA

        
        if args.num_gpu == 1 and torch.cuda.is_available():
            data = data.cuda()
            set_time(data, 0, 0, 0, batch_size=args.batch_size, device=device)
        try:
            with torch.no_grad():
                pred = model(data)
            #print("prediction",pred)
            all_pred.append(pred.detach().cpu())

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

    all_pred = torch.cat(all_pred).tolist() # TODO -> maybe list inside


    return all_pred


def main(args=None):
    printt("Starting Inference")

    if args is None:
        args = parse_args()
    torch.cuda.set_device(args.gpu)
    torch.hub.set_dir(args.torchhub_path)

    start_time = time.time()
    # load raw data
    data = load_data(args)
    data_params = data.data_params
    printt("finished loading raw data")

    # needs to be set if DataLoader does heavy lifting
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    # needs to be set if sharing resources
    if args.num_workers >= 1:
        torch.multiprocessing.set_sharing_strategy("file_system")

    
    # test mode: load up all replicates from checkpoint directory
    # and evaluate by sampling from reverse diffusion process
    if args.mode == "test":
        set_seed(args.seed)
        printt("running inference")
        test_scores = defaultdict(list)
    
        fold = 0
        # load and convert data to DataLoaders
        loaders = get_data(data, fold, args, for_reverse_diffusion=True)
        # print(loaders["test"].data)
        
        printt("finished creating data splits")
        # get model and load checkpoint, if relevant
        model = load_model(args, data_params, fold)
        model = to_cuda(model, args)

        model_confidence = load_model(args, data_params, fold,confidence_mode=True)
        model_confidence = to_cuda(model_confidence, args)
        printt("finished loading model")


        if args.wandb_sweep:
            def try_params():
                run = wandb.init()
                args.temp_sampling = wandb.config.temp_sampling
                args.temp_psi = wandb.config.temp_psi
                args.temp_sigma_data_tr = wandb.config.temp_sigma_data_tr
                args.temp_sigma_data_rot = wandb.config.temp_sigma_data_rot

                print(f'running run with: {args.temp_sampling, args.temp_psi, args.temp_sigma_data_tr, args.temp_sigma_data_rot}')
                printt('Running sequentially without confidence model')
                complex_rmsd_lt5 = []
                complex_rmsd_lt2 = []
                for i in tqdm(range(5)):
                    try:
                        samples_list = sample(loaders["val"], model, args)
                    except RuntimeError as e:
                        print(e)
                        print(traceback.format_exc())
                        raise e

                    meter = evaluate_all_rmsds(loaders["val"], samples_list)
                    ligand_rmsd_summarized, complex_rmsd_summarized, interface_rmsd_summarized = meter.summarize(verbose=False)
                    complex_rmsd_lt5.append(complex_rmsd_summarized['lt5'])
                    complex_rmsd_lt2.append(complex_rmsd_summarized['lt2'])
                    printt(f'Finished {i}-th sweep over the data')
                complex_rmsd_lt5 = np.array(complex_rmsd_lt5)
                complex_rmsd_lt2 = np.array(complex_rmsd_lt2)
                print(f'Average CRMSD < 5: {complex_rmsd_lt5.mean()}')
                print(f'Average CRMSD < 2: {complex_rmsd_lt2.mean()}')
                wandb.log({
                    'complex_rmsd_lt5': complex_rmsd_lt5.mean(),
                    'complex_rmsd_lt2': complex_rmsd_lt2.mean()
                })

            def try_params_with_confidence_model():
                wandb.login(key='36224bfda68d55fb01cd0404cb2b1186b6fd6d81', relogin=True)
                run = wandb.init()
                args.temp_sampling = wandb.config.temp_sampling
                args.temp_psi = wandb.config.temp_psi
                args.temp_sigma_data_tr = wandb.config.temp_sigma_data_tr
                args.temp_sigma_data_rot = wandb.config.temp_sigma_data_rot

                print(f'running run with confidence model with: {args.temp_sampling, args.temp_psi, args.temp_sigma_data_tr, args.temp_sigma_data_rot}')

                args.num_samples = 5

                # run reverse diffusion process
                try:
                    loaders_repeated,results = generate_loaders(loaders["val"],args) #TODO adapt sample size
                    
                    for i,loader in tqdm(enumerate(loaders_repeated), total=len(loaders_repeated)):
                        samples_list = sample(loader, model, args) #TODO: should work on data loader
                        samples_loader = DataLoader(samples_list,batch_size=args.batch_size)
                        pred_list = evaluate_confidence(model_confidence,samples_loader,args) # TODO -> maybe list inside
                        results[i]= results[i]+sorted(list(zip(samples_list,pred_list)),key=lambda x:-x[1]) 
                        printt("Finished Complex!")          
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    raise e

                printt(f'Finished run {args.run_name}')

                meter = evaluate_all_predictions(results)
                ligand_rmsd_summarized, complex_rmsd_summarized, interface_rmsd_summarized = meter.summarize(verbose=False)
                complex_rmsd_lt5 = complex_rmsd_summarized['lt5']
                complex_rmsd_lt2 = complex_rmsd_summarized['lt2']

                print(f'Average CRMSD < 5: {complex_rmsd_lt5}')
                print(f'Average CRMSD < 2: {complex_rmsd_lt2}')
                wandb.log({
                    'complex_rmsd_lt5': complex_rmsd_lt5,
                    'complex_rmsd_lt2': complex_rmsd_lt2
                })

            def try_actual_steps_with_confidence_model():
                wandb.login(key='36224bfda68d55fb01cd0404cb2b1186b6fd6d81', relogin=True)
                run = wandb.init()
                args.actual_steps = wandb.config.actual_steps

                print(f'Running with actual steps: {args.actual_steps}')

                args.num_samples = 10

                # run reverse diffusion process
                try:
                    loaders_repeated,results = generate_loaders(loaders["val"],args) #TODO adapt sample size
                    
                    for i,loader in tqdm(enumerate(loaders_repeated), total=len(loaders_repeated)):
                        samples_list = sample(loader, model, args) #TODO: should work on data loader
                        samples_loader = DataLoader(samples_list,batch_size=args.batch_size)
                        pred_list = evaluate_confidence(model_confidence,samples_loader,args) # TODO -> maybe list inside
                        results[i]= results[i]+sorted(list(zip(samples_list,pred_list)),key=lambda x:-x[1]) 
                        printt("Finished Complex!")          
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    raise e

                printt(f'Finished run {args.run_name}')

                meter = evaluate_all_predictions(results)
                ligand_rmsd_summarized, complex_rmsd_summarized, interface_rmsd_summarized = meter.summarize(verbose=False)
                complex_rmsd_lt5 = complex_rmsd_summarized['lt5']
                complex_rmsd_lt2 = complex_rmsd_summarized['lt2']

                print(f'Average CRMSD < 5: {complex_rmsd_lt5}')
                print(f'Average CRMSD < 2: {complex_rmsd_lt2}')
                wandb.log({
                    'complex_rmsd_lt5': complex_rmsd_lt5,
                    'complex_rmsd_lt2': complex_rmsd_lt2
                })


            # sweep_configuration = {
            #     'method': 'grid',
            #     'name': 'sweep',
            #     'metric': {'goal': 'maximize', 'name': 'complex_rmsd_lt2'},
            #     'parameters': 
            #     {
            #         'actual_steps': {'values': [30, 32, 34, 36, 38, 40]},
            #     }
            # }

            sweep_configuration = {
                'method': 'bayes',
                'name': 'sweep',
                'metric': {'goal': 'maximize', 'name': 'complex_rmsd_lt2'},
                'parameters': 
                {
                    'temp_sampling': {'max': 4.0, 'min': 0.0},
                    'temp_psi': {'max': 2.0, 'min': 0.0},
                    'temp_sigma_data_tr': {'max': 1.0, 'min': 0.0},
                    'temp_sigma_data_rot': {'max': 1.0, 'min': 0.0}
                }
            }
            sweep_id = wandb.sweep(sweep=sweep_configuration, project='DIPS optimize low temp with LRMSD conf model')

            wandb.agent(sweep_id, function=try_params_with_confidence_model, count=20)
            return

        if args.run_inference_without_confidence_model:
            printt('Running sequentially without confidence model')
            # loaders["test"].data = sorted(loaders["test"].data, key=lambda x:x['receptor_xyz'].shape[0] + x['ligand_xyz'].shape[0])

            # list_bs_32 = loaders["test"][0:64]
            # print(f'list_bs_32: {list_bs_32}')
            # list_bs_16 = loaders["test"][64:80]
            # print(f'list_bs_16: {list_bs_16}')
            # list_bs_8 = loaders["test"][80:88]
            # print(f'list_bs_8: {list_bs_8}')
            # list_bs_4 = loaders["test"][88:]
            # print(f'list_bs_4: {list_bs_4}')


            full_list = [loaders["val"]]
            complex_rmsd_lt5 = []
            complex_rmsd_lt2 = []
            time_to_load_data = time.time() - start_time
            print(f'time_to_load_data: {time_to_load_data}')
            start_time = time.time()
            for i in tqdm(range(1)):
                # print(f'bs: {32}')
                # samples_list_bs_32 = sample(list_bs_32, model, args, in_batch_size=32)
                # print(f'bs: {16}')
                # samples_list_bs_16 = sample(list_bs_16, model, args, in_batch_size=16)
                # print(f'bs: {8}')
                # samples_list_bs_8 = sample(list_bs_8, model, args, in_batch_size=8)
                # print(f'bs: {4}')
                # samples_list_bs_4 = sample(list_bs_4, model, args, in_batch_size=4)

                # samples_list = samples_list_bs_32 + samples_list_bs_16 + samples_list_bs_8 + samples_list_bs_4
                samples_list = sample(
                    loaders["val"], 
                    model, 
                    args, 
                    visualize_first_n_samples=args.visualize_n_val_graphs, 
                    visualization_dir=args.visualization_path,)
                full_list.append(samples_list)
                meter = evaluate_all_rmsds(loaders["val"], samples_list)
                ligand_rmsd_summarized, complex_rmsd_summarized, interface_rmsd_summarized = meter.summarize(verbose=True)
                complex_rmsd_lt5.append(complex_rmsd_summarized['lt5'])
                complex_rmsd_lt2.append(complex_rmsd_summarized['lt2'])
                printt(f'Finished {i}-th sweep over the data')

            end_time = time.time()
            print(f'Total time spent processing 5 times: {end_time-start_time}')
            print(f'time_to_load_data: {time_to_load_data}')

            complex_rmsd_lt5 = np.array(complex_rmsd_lt5)
            complex_rmsd_lt2 = np.array(complex_rmsd_lt2)
            print(f'Average CRMSD < 5: {complex_rmsd_lt5.mean()}')
            print(f'Average CRMSD < 2: {complex_rmsd_lt2.mean()}')
            dump_predictions(args,full_list)
            printt("Dumped data!!")
            return

        # data_list = BindingDataset(args, {}, apply_transform=False)
        # data_list.data = loaders["test"].data#[i] for i in range(len(loaders["test"].data))]
        # data_list.length = len(loaders["test"].data)

        # run reverse diffusion process
        print(f'args.temp_sampling: {args.temp_sampling}')
        
        loaders,results = generate_loaders(loaders["test"],args) #TODO adapt sample size
        
        for i,loader in tqdm(enumerate(loaders), total=len(loaders)):
            samples_list = sample(loader, model, args) #TODO: should work on data loader
            samples_loader = DataLoader(samples_list,batch_size=args.batch_size)
            pred_list = evaluate_confidence(model_confidence,samples_loader,args) # TODO -> maybe list inside
            results[i]= results[i]+sorted(list(zip(samples_list,pred_list)),key=lambda x:-x[1]) 
            printt("Finished Complex!")          
       

        printt(f'Finished run {args.run_name}')
        print(f'temp sampling, temp_psi, temp_sigma_data_tr, temp_sigma_data_rot: {args.temp_sampling, args.temp_psi, args.temp_sigma_data_tr, args.temp_sigma_data_rot}')
        print(f'filtering_model_path: {args.filtering_model_path}')

        end_time = time.time()
        print(f'Total time spent: {end_time-start_time}')
        meter = evaluate_all_predictions(results)
        #reverse_diffusion_metrics = evaluate_predictions(results)
        #printt(reverse_diffusion_metrics)
        
        dump_predictions(args,results)
        printt(f"Dumped data!! in {args.prediction_storage}")
        


        # log(test_scores, args.log_file, reduction=False)
        # end of all folds ========

def evaluate_all_predictions(results):
    ground_truth = [res[0][0] for res in results]
    best_pred = [res[1][0] for res in results]
    meter = evaluate_all_rmsds(ground_truth,best_pred)
    _ = meter.summarize()
    return meter


def evaluate_predictions(results):
    ground_truth = [res[0][0] for res in results]
    best_pred = [res[1][0] for res in results]
    eval_result = evaluate_pose(ground_truth,best_pred)
    rmsds = np.array(eval_result["rmsd"])
    reverse_diffusion_metrics = {'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
                                    'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds)),
                                    'rmsds_lt10': (100 * (rmsds < 10).sum() / len(rmsds)),
                                    'rmsds_mean': rmsds.mean(),
                                    'rmsds_median': np.median(rmsds)}
    return reverse_diffusion_metrics

def dump_predictions(args,results):
    with open(args.prediction_storage, 'wb') as f:
        pickle.dump(results, f)

def load_predictions(args):
    with open(args.prediction_storage, 'rb') as f:
        results = pickle.load(f)
    return results


def generate_loaders(loader,args):
    result = []
    data = loader.data
    ground_truth = []
    for d in data:
        #element = BindingDataset(args, {}, apply_transform=False)
        data_list = []
        
        for i in range(args.num_samples):
            data_list.append(copy.deepcopy(d))

        if args.mirror_ligand:
            #printt('Mirroring half of the complexes')
            for i in range(0, args.num_samples//2):
                e = data_list[i]["graph"]

                data = HeteroData()
                data["name"] = e["name"]

                data["receptor"].pos = e["ligand"].pos
                data["receptor"].x = e["ligand"].x

                data["ligand"].pos = e["receptor"].pos
                data["ligand"].x = e["receptor"].x

                data["receptor", "contact", "receptor"].edge_index = e["ligand", "contact", "ligand"].edge_index
                data["ligand", "contact", "ligand"].edge_index = e["receptor", "contact", "receptor"].edge_index

                # center receptor at origin
                center = data["receptor"].pos.mean(dim=0, keepdim=True)
                for key in ["receptor", "ligand"]:
                    data[key].pos = data[key].pos - center
                data.center = center  # save old center
                data["mirrored"] = True

                data_list[i]["graph"] = data

        element = BindingDataset(args, data_list, apply_transform=False)
        #element.data = data_list
        #element.length = args.num_samples
        # print(f'2: {element[2]}')
        # print(f'2_ligand: {element[2]["ligand"]}')
        # print(f'2_ligand_num_nodes: {element[2]["ligand"].num_nodes}')
        # print(f'0: {element[0]["ligand"].num_nodes}')
        
        result.append(element)
    
    for element in loader:
        ground_truth.append([(element,float("inf"))])
    return result,ground_truth




# def remove_label(loader):
#     if type(loader[0]) == tuple:
#         for i,element in enumerate(loader):
#             loader[i]=element
#     return loader

###############
# EXTRA CODE #ä
# for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
#     #if orig_complex_graph.name[0] not in limit_test:
#     #    continue
#     if filtering_model is not None and not (
#             filtering_args.use_original_model_cache or filtering_args.transfer_weights) and orig_complex_graph.name[
#         0] not in filtering_complex_dict.keys():
#         skipped += 1
#         print(
#             f"HAPPENING | The filtering dataset did not contain {orig_complex_graph.name[0]}. We are skipping this complex.")
#         continue
    
#     success = 0
#     while success == 0 or success == -1 or success == -2:
#         try:
#             success += 1
#             # TODO try to get the molecule directly from file without and check same results to avoid any kind of leak
#             data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]

#             pivot = None
#             if args.use_true_pivot:
#                 pivot = orig_complex_graph['ligand'].pos

#             randomize_position(data_list, score_model_args.no_torsion, args.no_random or args.no_random_pocket,
#                                score_model_args.tr_sigma_max if not args.pocket_knowledge else args.pocket_tr_max,
#                                args.pocket_knowledge, args.pocket_cutoff)

#             pdb = None
#             if args.save_visualisation:
#                 visualization_list = []
#                 for idx, graph in enumerate(data_list):
#                     lig = read_mol(args.data_dir, graph['name'][0], remove_hs=score_model_args.remove_hs)
#                     pdb = PDBFile(lig)
#                     pdb.add(lig, 0, 0)
#                     pdb.add((orig_complex_graph['ligand'].pos + orig_complex_graph.original_center).detach().cpu(), 1,
#                             0)
#                     pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
#                     visualization_list.append(pdb)
#             else:
#                 visualization_list = None

#             rec_path = os.path.join(args.data_dir, data_list[0]["name"][0],
#                                     f'{data_list[0]["name"][0]}_protein_processed.pdb')
#             if not os.path.exists(rec_path):
#                 rec_path = os.path.join(args.data_dir, data_list[0]["name"][0],
#                                         f'{data_list[0]["name"][0]}_protein_obabel_reduce.pdb')
#             rec = PandasPdb().read_pdb(rec_path)
#             rec_df = rec.df['ATOM']
#             receptor_pos = rec_df[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(
#                 np.float32) - orig_complex_graph.original_center.cpu().numpy()
#             receptor_pos = np.tile(receptor_pos, (N, 1, 1))
#             start_time = time.time()
#             if not args.no_model:
#                 if filtering_model is not None and not (
#                         filtering_args.use_original_model_cache or filtering_args.transfer_weights):
#                     filtering_data_list = [copy.deepcopy(filtering_complex_dict[orig_complex_graph.name[0]]) for _ in
#                                            range(N)]
#                 else:
#                     filtering_data_list = None

#                 data_list, confidence = sampling(data_list=data_list, model=model,
#                                                  inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
#                                                  tr_schedule=tr_schedule, rot_schedule=rot_schedule,
#                                                  tor_schedule=tor_schedule,
#                                                  device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
#                                                  no_random=args.no_random,
#                                                  ode=args.ode, visualization_list=visualization_list,
#                                                  confidence_model=filtering_model,
#                                                  filtering_data_list=filtering_data_list,
#                                                  filtering_model_args=filtering_model_args,
#                                                  asyncronous_noise_schedule=score_model_args.asyncronous_noise_schedule,
#                                                  t_schedule=t_schedule,
#                                                  batch_size=args.batch_size,
#                                                  no_final_step_noise=args.no_final_step_noise, pivot=pivot,
#                                                  svgd_weight=args.svgd_weight,
#                                                  svgd_repulsive_weight=args.svgd_repulsive_weight,
#                                                  svgd_only=args.svgd_only,
#                                                  svgd_rot_rel_weight=args.svgd_rot_rel_weight,
#                                                  svgd_tor_rel_weight=args.svgd_tor_rel_weight,
#                                                  temp_sampling=args.temp_sampling,
#                                                  temp_psi=args.temp_psi,
#                                                  temp_sigma_data=args.temp_sigma_data)

#                 if affinity_model is not None:
#                     if not (affinity_args.use_original_model_cache or affinity_args.transfer_weights):
#                         affinity_data_list = [copy.deepcopy(affinity_complex_dict[orig_complex_graph.name[0]]) for _ in
#                                               range(N)]
#                     else:
#                         affinity_data_list = None

#                     affinity_pred = compute_affinity(data_list=data_list, affinity_model=affinity_model,
#                                                      affinity_data_list=affinity_data_list,
#                                                      parallel=affinity_args.parallel,
#                                                      all_atoms=affinity_args.all_atoms, device=device, include_miscellaneous_atoms= hasattr(affinity_args, 'include_miscellaneous_atoms') and affinity_args.include_miscellaneous_atoms).cpu().item()
#                     true_affinities_list.append(affinities[orig_complex_graph.name[0]])
#                     pred_affinities_list.append(affinity_pred)

#                 if args.xtb:
#                     print(len(data_list), confidence[:, 0].shape)
#                     conf = confidence[:, 0].cpu().numpy()
#                     idx = np.argmax(conf)
#                     print(idx)
#                     optimize_complex(data_list[idx])
#             run_times.append(time.time() - start_time)
#             if score_model_args.no_torsion: orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

#             filterHs = torch.not_equal(data_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

#             if isinstance(orig_complex_graph['ligand'].orig_pos, list):
#                 orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

#             ligand_pos = np.asarray(
#                 [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in data_list])
#             orig_ligand_pos = np.expand_dims(
#                 orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(),
#                 axis=0)

#             try:
#                 mol = remove_all_hs(orig_complex_graph.mol[0])
#                 rmsd = get_symmetry_rmsd(mol, orig_ligand_pos[0], [l for l in ligand_pos])
#             except Exception as e:
#                 print("Using non corrected RMSD because of the error", e)
#                 rmsd = np.sqrt(((ligand_pos - orig_ligand_pos) ** 2).sum(axis=2).mean(axis=1))
#             rmsds_list.append(rmsd)
#             centroid_distance = np.linalg.norm(ligand_pos.mean(axis=1) - orig_ligand_pos.mean(axis=1), axis=1)
#             if confidence is not None and isinstance(filtering_args.rmsd_classification_cutoff, list):
#                 confidence = confidence[:, 0]
#             if confidence is not None:
#                 confidence = confidence.cpu().numpy()
#                 re_order = np.argsort(confidence)[::-1]
#                 print(orig_complex_graph['name'], ' rmsd', np.around(rmsd, 1)[re_order], ' centroid distance',
#                       np.around(centroid_distance, 1)[re_order], ' confidences ', np.around(confidence, 4)[re_order])
#                 confidences_list.append(confidence)
#             else:
#                 print(orig_complex_graph['name'], ' rmsd', np.around(rmsd, 1), ' centroid distance',
#                       np.around(centroid_distance, 1))
#             centroid_distances_list.append(centroid_distance)

#             if affinity_model is not None:
#                 print("true affinity", true_affinities_list[-1], "predicted affinity", pred_affinities_list[-1])

#             cross_distances = np.linalg.norm(receptor_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1)
#             min_cross_distances_list.append(np.min(cross_distances, axis=(1, 2)))
#             self_distances = np.linalg.norm(ligand_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1)
#             self_distances = np.where(np.eye(self_distances.shape[2]), np.inf, self_distances)
#             min_self_distances_list.append(np.min(self_distances, axis=(1, 2)))

#             base_cross_distances = np.linalg.norm(receptor_pos[:, :, None, :] - orig_ligand_pos[:, None, :, :], axis=-1)
#             base_min_cross_distances_list.append(np.min(base_cross_distances, axis=(1, 2)))

#             if args.save_visualisation:
#                 if confidence is not None:
#                     for rank, batch_idx in enumerate(re_order):
#                         visualization_list[batch_idx].write(
#                             f'{args.out_dir}/{data_list[batch_idx]["name"][0]}_{rank + 1}_{rmsd[batch_idx]:.1f}_{(confidence)[batch_idx]:.1f}.pdb')
#                 else:
#                     for rank, batch_idx in enumerate(np.argsort(rmsd)):
#                         visualization_list[batch_idx].write(
#                             f'{args.out_dir}/{data_list[batch_idx]["name"][0]}_{rank + 1}_{rmsd[batch_idx]:.1f}.pdb')
#             without_rec_overlap_list.append(1 if orig_complex_graph.name[0] in names_no_rec_overlap else 0)
#             names_list.append(orig_complex_graph.name[0])
#         except Exception as e:
#             print("Failed on", orig_complex_graph["name"], e)
#             failures += 1
#             success -= 2

if __name__ == "__main__":
    main()

