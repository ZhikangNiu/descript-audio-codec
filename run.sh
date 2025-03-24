export OMP_NUM_THREADS=1

DEBUG_MODE=true
EXP_NAME=$1
CONF_DIR=conf/vae
OUTPUT_DIR=ckpt/

if [ $DEBUG_MODE = true ]; then
    export CUDA_VISIBLE_DEVICES=0
    python scripts/train.py --args.load $CONF_DIR/$EXP_NAME --save_path $OUTPUT_DIR/DEBUG/$EXP_NAME
else
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    if [ -z "$PET_NPROC_PER_NODE" ]; then
        PET_NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
    fi
    torchrun --nproc_per_node $PET_NPROC_PER_NODE scripts/train.py \
    --args.load $CONF_DIR/$EXP_NAME
fi