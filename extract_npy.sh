#!/bin/bash

exp=$1

output_base=ldm_features/${exp}/
lt_root=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriTTS
ls_root=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriSpeech
tag=600k

for name in train-clean-100 train-clean-360 train-other-500; do
  python scripts/get_latent.py \
    --path "ckpts/svae/${exp}" \
    --input "$input_base/$name" \
    --output "$output_base/LibriTTS/$name" \
    --model_tag $tag
done

python scripts/get_latent.py --path "ckpts/svae/${exp}" --input ${ls_root}/test-clean --output "$output_base/LibriSpeech/test-clean" --model_tag $tag