import os, sys
import argparse
import subprocess
import multiprocessing
import datetime
import pickle
import random

import csv
import json
import itertools as it
from collections import ChainMap


# these arguments are shared among all experiments
COMMON_CONFIG = [
    {
        "seed": [0],
        "lr": [1e-4, 1e-5],
        "dropout": [0., 0.1],
        "weight_decay": [0., 1e-6],
        "epochs": [500],
        "patience": [50],
        "num_folds": [5],
        "num_conv_layers": [2, 4],
        "max_radius": [5., 10.]
    }
]

# we take the cartesian product of each individual set of args
# because some experiments require different settings
DATA_CONFIG = [
    #{
    #    "dataset": ["db5"],
    #    "data_file": ["data/db5.csv"],
    #    "data_path": ["/data/rsg/chemistry/rmwu/data/raw/binding/db5/"],
    #    "use_unbound": [None, ""],  # add flag or not (no value)
    #    # "None" will default to fine pose loader
    #    "pose_file": ["None", "data/db5_hpose.csv"],
    #    "pose_path": ["/data/rsg/chemistry/rmwu/data/processed/binding/energy/"],
    #},
    {
        "config_file": ["config/dips.yaml"],
        #"dataset": ["dips"],
        #"data_file": ["data/dips.csv"],
        #"data_path": ["/data/rsg/chemistry/rmwu/data/raw/binding/dips/pairs-pruned"],
    },
]
# these configure the models we run for each dataset
MODEL_CONFIG = [
    {
    #    "config_file": ["config/regression.json"],
    #    "use_all_atoms": [None],  # add flag or not (no value)
        "batch_size": [18]
    },
    #{
    #    "config_file": ["config/dips.yaml"],
    #},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Dispatcher to run all experiments")

    parser.add_argument("--num_gpu", type=int, default=6,
                        help="num gpus available to process. Default assume all "
                        "gpus < num are available.")
    parser.add_argument("--jobs_per_gpu", type=int, default=1,
                        help="number of jobs per GPU")
    parser.add_argument("--gpus_per_job", type=int, default=3,
                        help="number of GPUs per job")
    parser.add_argument("--log_dir", type=str,
                        default="/data/scratch/rmwu/tmp-runs/glue/dispatcher",
                        help="Directory for the log file.")
    parser.add_argument("--result_path", type=str,
                        default="results.csv",
                        help="path to store results table")
    parser.add_argument("--mode", type=str,
                        default="train",
                        help="train OR test")
    parser.add_argument("--rerun_experiments", action="store_true", default=False,
                        help="whether to rerun experiments with the same result "
                        "file location")

    return parser.parse_args()


def main():
    args = parse_args()

    # set up requirements
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    # compute cartesian product for each set of configurations
    common_configs = _chain_configs(COMMON_CONFIG)
    data_configs = _chain_configs(DATA_CONFIG)
    model_configs = _chain_configs(MODEL_CONFIG)

    all_configs = it.product(common_configs, data_configs, model_configs)
    all_configs = list(all_configs)
    random.shuffle(all_configs)  # avoid grouping large jobs on same GPUs

    # queues
    gpu_queues = {}
    for q in range(args.num_gpu * args.jobs_per_gpu // args.gpus_per_job):
        gpu_queues[q] = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    results = []
    indx = 0
    num_jobs = 0

    for config in all_configs:
        gpu_queues[indx].put(config)
        indx = (indx + 1) % len(gpu_queues)
        num_jobs += 1

    for gpu in range(len(gpu_queues)):
        job_queue = gpu_queues[gpu]
        gpu = gpu * args.gpus_per_job
        print("Start GPU worker {} with {} jobs".format(
            gpu, job_queue.qsize()))
        multiprocessing.Process(
                target=_worker, args=(gpu, job_queue, done_queue, args)).start()

    for _ in range(num_jobs):
        result_path, config = done_queue.get()

        try:
            with open(result_path, "r") as f:
                lines = f.readlines()  # load latest result (in case of re-runs)
                cur_res = json.loads(lines[-1])
                for d in config:
                    cur_res.update(d)
                results.append(cur_res)

        except Exception as e:
            print("Experiment at {} failed".format(result_path))
            print(e)
            continue

    with open(args.result_path, "w") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=results[0].keys())
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("Done")


def _chain_configs(configs):
    """
        @param configs list of configurations
    """
    all_configs = []
    for config in configs:
        # preserve order of configs
        keys = sorted(config)
        all_args = it.product(*(config[k] for k in keys))
        all_args_dict = [dict(zip(keys, c)) for c in all_args]
        all_configs.append(all_args_dict)

    return it.chain(*all_configs)  # flatten result


def _worker(gpu, queue, done_queue, args):
    while not queue.empty():
        configs = queue.get()
        if configs is None:
            return
        done_queue.put(_launch_experiment(gpu, configs, args))


def _launch_experiment(gpu, configs, args):

    log_stem, log_path, res_path = _get_log_name(configs, args)
    if not os.path.exists(log_stem):
        if args.mode == "train":
            os.makedirs(log_stem)
        else:
            print(f"{log_stem} DNE")
            return res_path, configs

    # >>> checkpoint_path
    cmd = ("python src/main.py --no_tqdm "
           f"--mode {args.mode} "
           f"--save_path {log_stem} "
           f"--checkpoint_path {log_stem} "  # recover in case of crash
           f"--tensorboard_path runs_dispatcher "
           f"--num_gpus={args.gpus_per_job} "
           f"--gpu={gpu} ")

    # add all keys to config
    cmd = _update_cmd(cmd, configs)

    # save output file path AFTER producing command
    configs[0]["log_path"] = log_stem

    # forward logs and stderr to logfile
    shell_cmd = "{} > {} 2>&1".format(cmd, log_path)
    print("Time {}, launched exp: {}".format(
        str(datetime.datetime.now()), shell_cmd))

    # if experiment has already been run, skip
    if not os.path.exists(res_path) or args.rerun_experiments:
        subprocess.call(shell_cmd, shell=True)

    if not os.path.exists(res_path):
        # running this process failed, alert me
        print("Dispatcher, Alert!",
                   "Job:[{}] has crashed! Check logfile at:[{}]".format(
                       cmd, log_path))

    return res_path, configs


def _get_log_name(configs, args):
    """
        @return log path
        @return results path
    """
    log_name = ""
    config = dict(ChainMap(*configs))
    for k, v in sorted(config.items()):
        if k in ["batch_size"]:
            continue
        # pretty formatting
        if type(v) is str:
            # trim file path keys
            if "/" in v and "config" not in v:
                continue
            if "/" in v:
                v = v.split("/")[-1]
            if "." in v:
                v = v.split(".")[0]
        if v is None:  # add flag without value
            log_name += "{}-".format(k)
        elif v == "":  # skip flag
            continue
        else:
            log_name += "{}={}-".format(k, v)

    log_name = log_name[:-1]  # remove last "-"
    log_stem = os.path.join(args.log_dir, log_name)

    log_path = f"{log_stem}/log.txt"
    res_path = f"{log_stem}/results.json"

    return log_stem, log_path, res_path


def _update_cmd(cmd, configs):
    """
        @param cmd str
        @param configs list of dicts
    """
    for config in configs:
        for k, v in config.items():
            if v is None:  # add flag without value
                cmd += "--{} ".format(k)
            elif v == "":  # skip flag
                continue
            else:
                cmd += "--{} {} ".format(k, v)

    return cmd


if __name__ == "__main__":
    main()

