# This script should contain run-specific information
# like GPU id, batch size, etc.
# Everything else should be specified in config.yaml

NUM_FOLDS=1  # number of seeds to try, default 5
SEED=0  # initial seed
CUDA=0  # will use GPUs from CUDA to CUDA + NUM_GPU - 1
NUM_GPU=1
BATCH_SIZE=2  # split across all GPUs

NAME="dips_esm"  # change to name of config file
RUN_NAME="large_model_dips" # should uniauely describe the current experiment
CONFIG="config/${NAME}.yaml"

SAVE_PATH="ckpts/${RUN_NAME}"
VISUALIZATION_PATH="visualization/${RUN_NAME}"

echo SAVE_PATH: $SAVE_PATH

python src/main.py \
    --mode "train" \
    --config_file $CONFIG \
    --run_name $RUN_NAME \
    --save_path $SAVE_PATH \
    --batch_size $BATCH_SIZE \
    --num_folds $NUM_FOLDS \
    --num_gpu $NUM_GPU \
    --gpu $CUDA --seed $SEED \
    --logger "wandb" \
    --project "DiffDock Tuning" \
    --visualize_n_val_graphs 0 \
    --visualization_path $VISUALIZATION_PATH \
    #--checkpoint_path $SAVE_PATH \
    #--debug True # load small dataset
    #--entity coarse-graining-mit \

# if you accidentally screw up and the model crashes
# you can restore training (including optimizer)
# by uncommenting --checkpoint_path