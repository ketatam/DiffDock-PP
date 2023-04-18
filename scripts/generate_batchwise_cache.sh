# This script should contain run-specific information
# like GPU id, batch size, etc.
# Everything else should be specified in config.yaml

NAME="dips_esm_batchwise_loading"  # change to name of config file
CONFIG="config/dips_esm_batchwise_loading.yaml"
CUDA=0


python scripts/generate_batchwise_cache.py \
    --config_file $CONFIG \
    --num_folds 0 \
    --gpu $CUDA

# if you accidentally screw up and the model crashes
# you can restore training (including optimizer)
# by uncommenting $SAVE_PATH
