#!/bin/bash
set -e

# 获取实验名称
exp_name=$1

if [ -z "$exp_name" ]; then
    echo "Error: Experiment name is required as an argument."
    echo "Usage: $0 <experiment_name>"
    exit 1
fi

# 测试数据路径
lt_test_other=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriTTS/test-other

# 日志文件夹路径
log_dir="./results/2gpu/${exp_name}/logs"

# 创建日志文件夹（如果不存在）
mkdir -p "${log_dir}"


# 循环处理每个模型标签
for tag in 10k 50k 100k 200k best; do
    echo "exp_name: runs/2gpu/${exp_name}"
    echo "output: ./results/2gpu/${exp_name}/$tag"
    echo "model_tag: ${tag}"

    if [ ! -d "runs/2gpu/${exp_name}/${tag}" ]; then
        echo "Path runs/2gpu/${exp_name}/${tag} does not exist. Skipping tag: ${tag}"
        continue
    fi

    # 运行 get_samples.py
    python scripts/get_samples.py \
        --path "runs/2gpu/${exp_name}" \
        --input "${lt_test_other}" \
        --output "./results/2gpu/${exp_name}/$tag" \
        --model_tag "$tag"

    # # 运行 evaluate.py，并将日志保存到文件
    # python scripts/evaluate.py \
    #     --input "${lt_test_other}" \
    #     --output "./results/2gpu/${exp_name}/$tag" \
    #     > "${log_dir}/${tag}_evaluate.log" 2>&1

    # echo "Evaluation log saved to ${log_dir}/${tag}_evaluate.log"
done


# 循环处理每个模型标签
for tag in 10k 50k 100k 200k best; do
    echo "exp_name: runs/2gpu/${exp_name}"
    echo "output: ./results/2gpu/${exp_name}/$tag"
    echo "model_tag: ${tag}"
    # 运行 evaluate.py，并将日志保存到文件
    if [ ! -d "./results/2gpu/${exp_name}/$tag" ]; then
        echo "Path ./results/2gpu/${exp_name}/$tag does not exist. Skipping tag: ${tag}"
        continue
    fi
    python scripts/evaluate.py \
        --input "${lt_test_other}" \
        --output "./results/2gpu/${exp_name}/$tag"
        
    python scripts/get_metrics.py ./results/2gpu/${exp_name}/$tag \
        > "${log_dir}/${tag}_evaluate.log" 2>&1
done


# python scripts/get_samples.py --path=runs/2gpu/320x_vae_dim32_grad10_kl5e-6 --input ${lt_test_other} --output ./results/2gpu/320x_vae_dim32_grad10_kl5e-6
# python scripts/get_samples.py --path=runs/2gpu/1600x_vae_dim32_grad10_kl5e-5 --input ${lt_test_other} --output ./results/2gpu/1600x_vae_dim32_grad10_kl5e-5
# python scripts/get_samples.py --path runs/2gpu/${exp_name} --input ${lt_test_other} --output ./results/2gpu/320x_vae_dim32_grad10 --model_tag 
# rm ./results/2gpu/320x_vae_dim32_grad10_kl5e-6/metrics.csv
# rm ./results/2gpu/1600x_vae_dim32_grad10_kl5e-5/metrics.csv
# python scripts/evaluate.py --input ${lt_test_other}  --output ./results/2gpu/320x_vae_dim32_grad10
# python scripts/evaluate.py --input ${lt_test_other}  --output ./results/2gpu/320x_vae_dim32_grad10_kl5e-6
# python scripts/evaluate.py --input ${lt_test_other}  --output ./results/2gpu/1600x_vae_dim32_grad10_kl5e-5
