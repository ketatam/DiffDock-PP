# This script should contain run-specific information
# like GPU id, batch size, etc.
# Everything else should be specified in config.yaml

# inference params
SEED=0
CUDA=2
NUM_GPU=2
BATCH_SIZE=40

NAME="dips_esm"
CONFIG="config/${NAME}.yaml"
CHECKPOINT="/data/scratch/rmwu/tmp-runs/glue/dips_esm"
SAVE_PATH=$CHECKPOINT
TEST_FOLD=3

echo SAVE_PATH: $SAVE_PATH

python src/main.py \
    --mode "test" \
    --config_file $CONFIG \
    --run_name $NAME \
    --save_path $SAVE_PATH \
    --checkpoint_path $CHECKPOINT \
    --batch_size $BATCH_SIZE \
    --test_fold $TEST_FOLD \
    --gpu $CUDA --seed $SEED \
    --num_gpu $NUM_GPU \

