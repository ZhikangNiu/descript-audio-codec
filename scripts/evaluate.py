import os
import csv
from dataclasses import dataclass
from pathlib import Path
import torchaudio

import argbind
import torch
from audiotools import AudioSignal
from audiotools import metrics
from audiotools.core import util
from train import losses
from tqdm import tqdm
from ecapa_tdnn import ECAPA_TDNN_SMALL
import torch.nn.functional as F
from zhon.hanzi import punctuation    
import zhconv
import string
from jiwer import compute_measures
    
punctuation_all = punctuation + string.punctuation
sr = 24000

@dataclass
class State:
    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    waveform_loss: losses.L1Loss
    sisdr_loss: losses.SISDRLoss

def load_asr_model(lang, ckpt_dir=""):
    if lang == "zh":
        from funasr import AutoModel

        model = AutoModel(
            model=os.path.join(ckpt_dir, "paraformer-zh"),
            disable_update=True,
        )  # following seed-tts setting
    elif lang == "en":
        from faster_whisper import WhisperModel

        model_size = "large-v3" if ckpt_dir == "" else ckpt_dir
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    return model

@torch.no_grad()
def get_sim_score(wav1,wav2,model):
    resample1 = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    resample2 = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    wav1 = resample1(wav1.audio_data).squeeze(0).cuda()
    wav2 = resample2(wav2.audio_data).squeeze(0).cuda()
    emb1 = model(wav1)
    emb2 = model(wav2)
    sim = F.cosine_similarity(emb1, emb2)[0].item()
    return sim

def get_wer_score(gen_wav,truth,asr_model,lang):
    if lang == "zh":
        res = asr_model.generate(input=gen_wav, batch_size_s=300, disable_pbar=True)
        hypo = res[0]["text"]
        hypo = zhconv.convert(hypo, "zh-cn")  # 繁简转换，可选
    elif lang == "en":
        segments, _ = asr_model.transcribe(gen_wav, beam_size=5, language="en")
        hypo = ""
        for segment in segments:
            hypo += " " + segment.text
    else:
        raise NotImplementedError("Only 'zh' or 'en' are supported.")

    for p in punctuation_all:
        truth = truth.replace(p, "")
        hypo = hypo.replace(p, "")

    truth = truth.replace("  ", " ")
    hypo = hypo.replace("  ", " ")

    if lang == "zh":
        # 中文可以将每个汉字用空格分隔，以便 jiwer 的统计
        truth = " ".join(list(truth))
        hypo = " ".join(list(hypo))
    elif lang == "en":
        # 英文转小写
        truth = truth.lower()
        hypo = hypo.lower().lstrip()

    measures = compute_measures(truth, hypo)
    wer = measures["wer"]

    return wer,truth,hypo

def get_metrics(signal_path, recons_path, state,asr_model,sim_model,lang):
    output = {}
    signal = AudioSignal(signal_path)
    recons = AudioSignal(recons_path)
    # for sr in [24000, 44100]:
    x = signal.clone().resample(sr)
    y = recons.clone().resample(sr)
    if x.signal_length != y.signal_length:
        min_length = min(x.signal_length,y.signal_length)
        x = x[:,:,:min_length]
        y = y[:,:,:min_length]
    # import ipdb;ipdb.set_trace()
    k = "24k" if sr == 24000 else "44k"
    output["name"] = signal.path_to_file.stem
    sim_score = get_sim_score(x,y,sim_model)
    truth_text_path = signal_path.with_suffix(".original.txt")
    with open(truth_text_path,"r") as f:
        truth = f.readline().strip()
    wer_score,truth,hypo = get_wer_score(recons_path,truth,asr_model=asr_model,lang=lang)
    output.update(
        {
            f"mel-{k}": state.mel_loss(x, y),
            f"stft-{k}": state.stft_loss(x, y),
            f"waveform-{k}": state.waveform_loss(x, y),
            f"sisdr-{k}": state.sisdr_loss(x, y),
            f"pseq-{k}" : metrics.quality.pesq(x,y),
            f"stoi-{k}" : metrics.quality.stoi(x,y),
            f"sim-{k}": sim_score,
            f"wer-{k}": wer_score,
            f"truth": truth,
            f"hypo": hypo
            # f"visqol-speech-{k}": metrics.quality.visqol(x, y, "speech"),
        }
    )
    # output.update(signal.metadata)
    return output


@argbind.bind(without_prefix=True)
@torch.inference_mode()
def evaluate(
    input: str = "samples/input", # gt
    output: str = "samples/output", # recon
    lang: str = "en"
):
    asr_ckpt_dir = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/faster-whisper-large-v3"
 
    # --------------------------- WER ---------------------------
    wavlm_ckpt_dir = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/wavlm_large_finetune.pth"
    
    sim_model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(wavlm_ckpt_dir, weights_only=True, map_location=lambda storage, loc: storage)
    sim_model.load_state_dict(state_dict["model"], strict=False)
    sim_model = sim_model.cuda()
    sim_model.eval()
    asr_model = load_asr_model(lang,asr_ckpt_dir)

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    sisdr_loss = losses.SISDRLoss()

    state = State(
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        sisdr_loss=sisdr_loss,
    )

    audio_files = util.find_audio(input)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    
    def record(o, writer):
        for k, v in o.items():
            if torch.is_tensor(v):
                o[k] = v.item()
        writer.writerow(o)
        return o

    all_results = []
    for i in tqdm(range(len(audio_files))):
        try:
            file_results = get_metrics(audio_files[i], output / audio_files[i].name, state,asr_model=asr_model,lang=lang,sim_model=sim_model)
            all_results.append(file_results)
        except FileNotFoundError as e:
            print(f"Not found {output / audio_files[i].name}")
    

    with open(output / "metrics.csv", "w") as csvfile:
        keys = list(all_results[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for line in all_results:
            record(line, writer)

if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate()
