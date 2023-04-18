# This script should contain run-specific information
# like GPU id, batch size, etc.
# Everything else should be specified in config.yaml

NUM_FOLDS=1  # number of seeds to try, default 5
SEED=0  # initial seed
CUDA=0  # will use GPUs from CUDA to CUDA + NUM_GPU - 1
NUM_GPU=1
BATCH_SIZE=16  # split across all GPUs

NAME="dips_esm_confidence"  # change to name of config file
RUN_NAME="dips_confidence_model"
CONFIG="config/${NAME}.yaml"

# you may save to your own directory
SAVE_PATH="ckpts/${RUN_NAME}"
VISUALIZATION_PATH="visualization/${RUN_NAME}"

SAMPLES_DIRECTORY="datasets/DIPS/confidence_full"

echo SAVE_PATH: $SAVE_PATH

python src/main_confidence.py \
    --mode "train" \
    --config_file $CONFIG \
    --run_name $RUN_NAME \
    --save_path $SAVE_PATH \
    --batch_size $BATCH_SIZE \
    --num_folds $NUM_FOLDS \
    --num_gpu $NUM_GPU \
    --gpu $CUDA --seed $SEED \
    --logger "wandb" \
    --project "Confidence model" \
    --samples_directory $SAMPLES_DIRECTORY \
    #--checkpoint_path $SAVE_PATH \
    #--debug True # load small dataset
    #--entity coarse-graining-mit \

# if you accidentally screw up and the model crashes
# you can restore training (including optimizer)
# by uncommenting --checkpoint_path

