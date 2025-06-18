export OMP_NUM_THREADS=1

DEBUG_MODE=false
EXP_NAME=$1
CONF_DIR=conf/svae
OUTPUT_DIR=ckpts/

# 检查是否有CUDA_VISIBLE_DEVICES环境变量，如果没有，则设置为0,1,2,3,4,5,6,7
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

if [ $DEBUG_MODE = true ]; then
    export CUDA_VISIBLE_DEVICES=0
    python -m debugpy --wait-for-client --listen 5678 scripts/train.py --args.load $CONF_DIR/$EXP_NAME --save_path $OUTPUT_DIR/DEBUG/$EXP_NAME
else

    if [ -z "$PET_NPROC_PER_NODE" ]; then
        PET_NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
    fi
    torchrun --nproc_per_node $PET_NPROC_PER_NODE scripts/train.py \
    --args.load $CONF_DIR/$EXP_NAME
fi