#!/bin/bash
set -e
export OMP_NUM_THREADS=1
exp=$1
nnodes=1
nproc_per_node=8

output_base="ldm_features/${exp}/"
input_base="/vepfs/user/tts/speech/LibriTTS"
tag=600k
ckpt_name="ckpts/svae/${exp}"

for name in train-clean-100 train-clean-360 train-other-500; do
  torchrun \
  --nnodes $nnodes \
  --nproc_per_node $nproc_per_node \
  scripts/extract_latent_features.py \
  --path $ckpt_name \
  --input "$input_base/$name" \
  --output "$output_base/LibriTTS/$name" \
  --model_tag $tag
done

torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
scripts/extract_latent_features.py \
--path $ckpt_name \
--input /vepfs/user/tts/speech/LibriSpeech/test-clean \
--output "$output_base/LibriSpeech/test-clean" \
--model_tag $tag