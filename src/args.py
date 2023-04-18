import os
import yaml
import argparse

from utils import printt


def parse_args():
    parser = argparse.ArgumentParser("DiffDock for proteins")

    # configuration
    parser.add_argument("--debug",
                        type=bool, default=False,
                        help="Set flag true to load smaller dataset")
    parser.add_argument("--config_file",
                        type=str, default=None,
                        help="YAML file")
    parser.add_argument("--args_file",
                        type=str, default="args.yaml",
                        help="Dump arguments for reproducibility")
    parser.add_argument("--log_file",
                        type=str, default="results.yaml",
                        help="Save results here")
    parser.add_argument("--tuned_param",
                        type=str, default="",
                        help="Which hyperparamter to tune.")

    # ======== data ========
    # include path to pdb protein files, splits
    parser.add_argument("--data_file",
                        type=str, default="",
                        help="Includes path to PDB files, splits")
    parser.add_argument("--recache",
                        action="store_true",
                        help="Flag to recache the processed input. Does not include graph and lm cache.")
    parser.add_argument("--no_graph_cache",
                        action="store_true",
                        help="Flag to disable caching of graphs")
    parser.add_argument("--no_lm_cache",
                        action="store_true",
                        help="Flag to disable caching of ESM")

    parser.add_argument("--data_path",
                        type=str, default="",
                        help="Root to raw data directory")
    parser.add_argument("--pose_path",
                        type=str, default="data",
                        help="Root to raw test poses directory")
    parser.add_argument("--save_path",
                        type=str, default="",
                        help="Root to model checkpoints")
    parser.add_argument("--torchhub_path",
                        type=str, default="torchhub",
                        help="Root to torch hub cache")
    parser.add_argument("--tensorboard_path",
                        type=str, default="runs",
                        help="Tensorboard directory")

    # specify data loading parameters
    parser.add_argument("--dataset",
                        choices=["dips", "db5", "toy"],
                        type=str, default="db5",
                        help="")
    parser.add_argument("--use_unbound",
                        action="store_true",
                        help="Bound or unbound for DB5")

    parser.add_argument("--resolution", default="residue",
                        choices=["residue", "backbone", "atom"],
                        help="resolution of individual points")

    parser.add_argument("--use_orientation_features", action="store_true", default=False,
                        help="If set, use orientation features for the edges as in EquiDock")

    # data loading
    parser.add_argument("--max_poses",
                        type=int, default=100,
                        help="maximum number of poses to load for eval")

    parser.add_argument("--num_workers",
                        type=int, default=0,
                        help="DataLoader workers")
    parser.add_argument("--batch_size",
                        type=int, default=10,
                        help="number of protein complexes per batch")
    # logging
    parser.add_argument("--logger",
                        type=str, choices=["wandb", "tensorboard"],
                        default="tensorboard",
                        help="Which logger to use, wandb or tensorboard")
    parser.add_argument("--run_name",
                        type=str, default=None,
                        help=("(optional) tensorboard folder, aka 'comment' "
                              "field. used for dispatcher. If wandb is used, the name of the run."))
    parser.add_argument("--wandb_run_name",
                        type=str, default=None,
                        help=("(optional) Run name for wandb "
                              "If not provided, then run_name will be used."))
    parser.add_argument("--log_frequency",
                        type=int, default=10,
                        help="log every [n] batches")
    parser.add_argument("--project",
                        type=str, default=None,
                        help="For wandb logger, the name of the project where you're sending the new run.")
    parser.add_argument("--entity",
                        type=str, default=None,
                        help="For wandb logger, a username or team name where you're sending runs.")
    parser.add_argument("--group",
                        type=str, default=None,
                        help="For wandb logger, a group to organize individual runs into a larger experiment.")
    parser.add_argument("--visualize_n_val_graphs",
                        type=int, default=5,
                        help="How many of validation graphs should be visualized.")
    parser.add_argument("--visualization_path",
                        type=str, default="./visualization",
                        help="Where to save visualizations.")

    # data processing
    parser.add_argument('--receptor_radius', type=float, default=30,
                        help='Cutoff on distances for receptor edges')
    parser.add_argument('--c_alpha_max_neighbors', type=int, default=10,
                        help='Max number of neighbors for each residue')
    parser.add_argument('--atom_radius', type=float, default=5,
                        help='Cutoff on distances for atom connections')
    parser.add_argument('--atom_max_neighbors', type=int, default=8,
                        help='Max number of atom neighbours for receptor')
    parser.add_argument('--matching_popsize', type=int, default=20,
                        help='Differential evolution popsize parameter in matching')
    parser.add_argument('--matching_maxiter', type=int, default=20,
                        help='Differential evolution maxiter parameter in matching')
    parser.add_argument('--max_lig_size', type=int, default=None,
                        help='Max number of heavy atoms in ligand')
    parser.add_argument('--remove_hs', action='store_true', default=False,
                        help='remove Hs')
    parser.add_argument('--multiplicity', type=int, default=1, help='multiplicity parameter for debugging')

    # ====== training ======
    parser.add_argument("--mode",
                        choices=["train", "test"],
                        type=str, default="train",
                        help="Training or inference")

    parser.add_argument("--num_folds",
                        type=int, default=5,
                        help="Number of different seeds = cv folds")
    parser.add_argument("--test_fold",
                        type=int, default=0,
                        help="Fold to use for inference")
    parser.add_argument("--epochs",
                        type=int, default=200,
                        help="Max epochs to train")
    parser.add_argument("--patience",
                        type=int, default=10,
                        help="Lack of validation improvement for [n] epochs")
    parser.add_argument("--metric",
                        type=str, default="loss",
                        help="For printing only")

    parser.add_argument("--gpu",
                        type=int, default=0,
                        help="GPU id")
    parser.add_argument("--num_gpu",
                        type=int, default=1,
                        help="number of GPU for DataParallel")
    parser.add_argument("--seed",
                        type=int, default=0,
                        help="Initial seed")

    parser.add_argument("--save_pred",
                        action="store_true",
                        help="Save predictions on test set")
    parser.add_argument("--no_tqdm",
                        action="store_true",
                        help="Set to True if running dispatcher")
    parser.add_argument("--save_model_every",
                        type=int, default=10,
                        help="Frequency (in epochs) of saving the latest model")

    # ====== inference ======
    parser.add_argument("--num_steps",
                        type=int, default=20,
                        help="Number of denoising steps")
    parser.add_argument("--actual_steps",
                        type=int, default=40,
                        help="Number of actual denoising steps. The intuition is to cut the last steps short because those tend to overfit.")
    parser.add_argument("--ode",
                        action="store_true",
                        help="Use ODE for inference")
    parser.add_argument("--no_random",
                        action="store_true",
                        help="Use no randomness in reverse diffusion")
    parser.add_argument("--no_final_noise",
                        action="store_true",
                        help="Use no noise in the final step of "
                             "reverse diffusion")
    parser.add_argument('--val_inference_freq',
                        type=int,
                        default=5,
                        help='Frequency of epochs for which to run expensive inference on val data')
    parser.add_argument('--num_inference_complexes',
                        type=int,
                        default=None,
                        help='Number of complexes for which inference is run every val/train_inference_freq epochs (None will run it on all)')
    parser.add_argument('--num_inference_complexes_train_data',
                        type=int,
                        default=None,
                        help='Number of complexes for which inference is run every val/train_inference_freq epochs (None will run it on all)')
    parser.add_argument("--sample_train",
                        action="store_true",
                        help="Whether to run inference on training data. Useful for DB5.")


    # ======== model =======
    parser.add_argument("--model_type",
                        choices=["diffusion"],
                        type=str, default="diffusion")
    parser.add_argument("--ebd_type",
                        choices=["continuous", "discrete"],
                        type=str, default="continuous")
    parser.add_argument("--encoder_type",
                        choices=["e3nn"],
                        type=str, default="e3nn",
                        help="protein encoder")
    # (optional)
    parser.add_argument("--checkpoint_path",
                        type=str, default=None,
                        help="Checkpoint for entire model for test/finetune")

    # Embeddings
    parser.add_argument("--hidden_size",
                        type=int, default=64,
                        help="Hidden layer representation size")
    parser.add_argument("--dropout",
                        type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--knn_size",
                        type=int, default=20,
                        help="max kNN size for edge construction")

    # E3NN
    parser.add_argument("--num_conv_layers",
                        type=int, default=2,
                        help="Number of interaction layers")
    parser.add_argument("--max_radius",
                        type=float, default=5.,
                        help="Radius cutoff for geometric graph (A)")
    parser.add_argument("--scale_by_sigma", action="store_true", default=True,
                        help="Whether to normalise the score")
    parser.add_argument("--ns",
                        type=int, default=16,
                        help="Number of hidden features per node of order 0")
    parser.add_argument("--nv",
                        type=int, default=4,
                        help="Number of hidden features per node of order >0")

    parser.add_argument("--dist_embed_dim",
                        type=int, default=32,
                        help="Embedding size for the distance")
    parser.add_argument("--cross_dist_embed_dim",
                        type=int, default=32,
                        help="Embeddings size for the cross distance")
    parser.add_argument("--lm_embed_dim",
                        type=int, default=0,
                        help="0 or 1280 for ESM2")

    parser.add_argument("--no_batch_norm", action="store_true", default=False,
                        help="If set, it removes the batch norm")
    parser.add_argument("--use_second_order_repr",
                        action="store_true", default=False,
                        help="Whether to use only up to first order representations or also second")
    parser.add_argument("--cross_max_dist",
                        type=float, default=80,
                        help="Max cross distance in case not dynamic")
    parser.add_argument("--dynamic_max_cross",
                        action="store_true", default=False,
                        help="Whether to use the dynamic distance cutoff")
    parser.add_argument("--cross_cutoff_weight",
                        type=float, default=3,
                        help="The weight that multpilies tr_s in case dynamic. Dynamic cross cutoff is of the form tr_s * weight + bias")
    parser.add_argument("--cross_cutoff_bias",
                        type=float, default=40,
                        help="The bias that gets added to tr_s * weight in case dynamic. Dynamic cross cutoff is of the form tr_s * weight + bias")
    parser.add_argument("--embedding_type",
                        type=str, default="sinusoidal",
                        help="Type of diffusion time embedding")
    parser.add_argument("--sigma_embed_dim",
                        type=int, default=32,
                        help="Size of the embedding of the diffusion time")
    parser.add_argument("--embedding_scale",
                        type=int, default=10000,
                        help="Parameter of the diffusion time embedding")

    # ==== optimization ====
    # loss term weights
    parser.add_argument("--score_loss_weight", default=0.,
                        type=float,
                        help="Loss weight for score matching")
    parser.add_argument("--energy_loss_weight", default=1.,
                        type=float,
                        help="Loss weight for energy margin")

    # optimizer
    parser.add_argument("--lr",
                        type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay",
                        type=float, default=1e-6,
                        help="L2 regularization weight")

    # diffusion
    parser.add_argument("--tr_weight",
                        type=float, default=0.33,
                        help="Weight of translation loss")
    parser.add_argument("--rot_weight",
                        type=float, default=0.33,
                        help="Weight of rotation loss")
    parser.add_argument("--tor_weight",
                        type=float, default=0.33,
                        help="Weight of torsional loss")

    parser.add_argument("--rot_s_min",
                        type=float, default=0.1,
                        help="Min sigma for rotational component")
    parser.add_argument("--rot_s_max",
                        type=float, default=1.65,
                        help="Max sigma for rotational component")
    parser.add_argument("--tr_s_min",
                        type=float, default=0.1,
                        help="Min sigma for translational component")
    parser.add_argument("--tr_s_max",
                        type=float, default=30,
                        help="Max sigma for translational component")
    parser.add_argument("--tor_s_min",
                        type=float, default=0.0314,
                        help="Min sigma for torsional component")
    parser.add_argument("--tor_s_max",
                        type=float, default=3.14,
                        help="Max sigma for torsional component")
    parser.add_argument("--no_torsion", action="store_true", default=False,
                        help="If set only rigid matching")

    # confidence model
    parser.add_argument('--rmsd_prediction', action='store_true', 
                        default=False, 
                        help='If true, use regression against RMSD values')
    parser.add_argument('--rmsd_classification_cutoff', 
                        type=float, default=5, 
                        help='RMSD value below which a prediction is considered a postitive.')
    parser.add_argument("--generate_n_predictions", type=int, default=7,
                        help="For generating samples, how many predictions should be generated per samples.")
    parser.add_argument("--samples_directory", type=str, default="",
                        help="Directory in which sampels should be saved & loaded for confidence model.")
    parser.add_argument('--use_randomized_confidence_data', action='store_true', 
                        default=False, 
                        help='If true, randomly generated data for training the confidence mdoel. Else, use data generated by score model')
    parser.add_argument('--filtering_model_path', 
                        default=None, type=str,
                        help='path to filtering model')
    parser.add_argument('--score_model_path',type=str, 
                        default=None, 
                        help='path to score model')
    
    parser.add_argument('--num_samples',type=int, 
                        default=None, 
                        help='number of samples in inference')

    parser.add_argument('--mirror_ligand',type=bool, 
                    default=False, 
                    help='number of samples in inference')
    
    parser.add_argument('--prediction_storage',type=str, 
                default=False, 
                help='output path to predictions')

    parser.add_argument("--temp_sampling", type=float, default=1.0, help="")
    parser.add_argument("--temp_psi", type=float, default=0.0, help="")
    parser.add_argument("--temp_sigma_data_tr", type=float, default=0.5, help="")
    parser.add_argument("--temp_sigma_data_rot", type=float, default=0.5, help="")
                

    parser.add_argument('--run_inference_without_confidence_model', action='store_true', 
                        default=False, 
                        help='If true, generate samples on the data, one full sweep at a time. Used to estimate raw performance without confidence_model')

    parser.add_argument('--use_complex_rmsd', action='store_true', 
                        default=False, 
                        help='If True, use complex RMSD values to train the confidence model.')

    parser.add_argument('--use_interface_rmsd', action='store_true', 
                        default=False, 
                        help='If True, use Interface RMSD values to train the confidence model.')

    parser.add_argument('--wandb_sweep', action='store_true', 
                        default=False, 
                        help='If True, use wandb sweep to optimize hyperparams for low-temp sampling')

    args = parser.parse_args()
    process_args(args)
    return args


def process_args(args):
    """
        This function does a couple of nice things:
        1)  load any arguments specified in config_file
        2)  set default save_path and args_path if checkpoint provided
        3)  load any remaining arguments saved from checkpoint,
            if applicable. config_file takes precedence over saved
            args from checkpoint directory
    """

    # used for dispatcher only (bash script auto-formats to config)
    ## process run_name
    if args.run_name is None:
        args.run_name = args.save_path.split("/")[-1]

    # load configuration = override specified values
    ## load config_file
    if args.config_file is not None:
        with open(args.config_file) as f:
            config = yaml.safe_load(f)
        override_args(args, config)

    # default checkpoint_path to save_path if mode == "test"
    #TODO: change back
    # if args.mode == "test":
    #     if args.checkpoint_path is None:
    #         args.checkpoint_path = args.save_path
    #     elif args.save_path == "":
    #         args.save_path = args.checkpoint_path

    # prepend output root
    if args.checkpoint_path is not None:
        args.args_file = os.path.join(args.checkpoint_path, args.args_file)
    else:
        args.args_file = os.path.join(args.save_path, args.args_file)
    args.log_file  = os.path.join(args.save_path, args.log_file)

    # finally load all saved parameters
    if args.checkpoint_path is not None:
        if not os.path.exists(args.checkpoint_path):
            printt("invalid checkpoint_path", args.checkpoint_path)
        if os.path.exists(args.args_file):
            with open(args.args_file) as f:
                saved_config = yaml.safe_load(f)
        print("here",args.args_file)
        # do not overwrite certain args
        # outer_key is 'data', 'mode', ... inner_key is the correct key
        k_to_skip = [inner_key for outer_key in config.keys() for inner_key in config[outer_key].keys() ]
        # k_to_skip = list(config.keys())  # config takes precedence BUG!
        k_to_skip.extend(["checkpoint_path", "save_path",
                          "gpu", "mode", "test_fold",
                          "batch_size", "debug", "data_file", "num_gpu", "generate_n_predictions",
                          "samples_directory", "logger", "project", "config_file"])
        for k in k_to_skip:
            if k in saved_config:
                del saved_config[k]
        override_args(args, saved_config)


def override_args(args, config):
    """
        Recursively copy over config to args
    """
    for k,v in config.items():
        if type(v) is dict:
            override_args(args, v)
        else:
            args.__dict__[k] = v
    return args

