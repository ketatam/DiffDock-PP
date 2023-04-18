# This script should contain run-specific information
# like GPU id, batch size, etc.
# Everything else should be specified in config.yaml

SEED=0  # initial seed
CUDA=0  # will use GPUs from CUDA to CUDA + NUM_GPU - 1
NUM_GPU=2
BATCH_SIZE=16  # split across all GPUs

NAME="dips_esm_batchwise_loading"  # change to name of config file
RUN_NAME="dips_large_model_faster"
CONFIG="config/${NAME}.yaml"

# you may save to your own directory
SAVE_PATH="ckpts/${RUN_NAME}"

SAMPLES_DIRECTORY="datasets/DIPS/confidence_full"

echo SAVE_PATH: $SAVE_PATH

python src/main_generate_samples.py \
    --config_file $CONFIG \
    --run_name $RUN_NAME \
    --save_path $SAVE_PATH \
    --batch_size $BATCH_SIZE \
    --num_folds 0 \
    --num_gpu $NUM_GPU \
    --gpu $CUDA \
    --seed $SEED \
    --checkpoint_path $SAVE_PATH \
    --samples_directory $SAMPLES_DIRECTORY \
    --generate_n_predictions 4

# if you accidentally screw up and the model crashes
# you can restore training (including optimizer)
# by uncommenting $SAVE_PATH
