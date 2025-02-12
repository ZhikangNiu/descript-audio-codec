set -e
pip install --upgrade faster-whisper==1.1.1 && pip install --upgrade ctranslate2==4.5.0
if [ ! -d "$HOME/.cache/torch" ]; then
    mkdir -p "$HOME/.cache/torch"
fi
cp -r /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/hub ~/.cache/torch
cp -r /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/s3prl ~/.cache
gpu_nums=4 # default 4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`python3 -c 'import os; import torch; print(os.path.dirname(torch.__file__) +"/lib")'`

# python scripts/eval_recon_wer.py -l en -g /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/descript-audio-codec/LibriSpeech/test-clean/50hz_feat -p /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriSpeech/test-clean -n 1 --local -t sim

python scripts/eval_recon_wer.py -l en -g /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriSpeech/test-clean -p /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriSpeech/test-clean -n 1 --local -t sim