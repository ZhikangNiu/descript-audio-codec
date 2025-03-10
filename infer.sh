#!/bin/bash
set -e
pip install --upgrade faster-whisper==1.1.1 && pip install --upgrade ctranslate2==4.5.0
if [ ! -d "$HOME/.cache/torch" ]; then
    mkdir -p "$HOME/.cache/torch"
fi
cp -r /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/hub ~/.cache/torch
cp -r /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/s3prl ~/.cache
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python3 -c 'import os; import torch; print(os.path.dirname(torch.__file__) +"/lib")'`

# 获取实验名称
exp_name=$1

if [ -z "$exp_name" ]; then
    echo "Error: Experiment name is required as an argument."
    echo "Usage: $0 <experiment_name>"
    exit 1
fi

# 测试数据路径
lt_test_other=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/descript-audio-codec/test_subset

# 日志文件夹路径
log_dir="./results/vae_exp_large/subset/${exp_name}/logs"

# 创建日志文件夹（如果不存在）
mkdir -p "${log_dir}"


# 循环处理每个模型标签
for tag in 200k 300k 400k 500k 600k best latest; do
    echo "exp_name: vae_exp_large/${exp_name}"
    echo "output: ./results/vae_exp_large/subset/${exp_name}/$tag"
    echo "model_tag: ${tag}"

    if [ ! -d "vae_exp_large/${exp_name}/${tag}" ]; then
        echo "Path vae_exp_large/${exp_name}/${tag} does not exist. Skipping tag: ${tag}"
        continue
    fi

    # 运行 get_samples.py
    python scripts/get_samples.py \
        --path "vae_exp_large/${exp_name}" \
        --input "${lt_test_other}" \
        --output "./results/vae_exp_large/subset/${exp_name}/$tag" \
        --model_tag "$tag"
done


# # 循环处理每个模型标签
for tag in 200k 300k 400k 500k 600k best latest; do
    echo "exp_name: vae_exp_large/${exp_name}"
    echo "output: ./results/vae_exp_large/subset/${exp_name}/$tag"
    echo "model_tag: ${tag}"
    # 运行 evaluate.py，并将日志保存到文件
    if [ ! -d "vae_exp_large/${exp_name}/${tag}" ]; then
        echo "Path vae_exp_large/${exp_name}/${tag} does not exist. Skipping tag: ${tag}"
        continue
    fi
    # python -m debugpy --listen 5678 --wait-for-client 
    python scripts/evaluate.py \
        --input "${lt_test_other}" \
        --output "./results/vae_exp_large/subset/${exp_name}/$tag" \

    python scripts/get_metrics.py ./results/vae_exp_large/subset/${exp_name}/$tag \
        > "${log_dir}/${tag}_evaluate.log" 2>&1
done
