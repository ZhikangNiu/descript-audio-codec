import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools import metrics
from audiotools.core import util
from train import losses
from tqdm import tqdm

sr = 24000
@dataclass
class State:
    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    waveform_loss: losses.L1Loss
    sisdr_loss: losses.SISDRLoss


def get_metrics(signal_path, recons_path, state):
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
    output.update(
        {
            f"mel-{k}": state.mel_loss(x, y),
            f"stft-{k}": state.stft_loss(x, y),
            f"waveform-{k}": state.waveform_loss(x, y),
            f"sisdr-{k}": state.sisdr_loss(x, y),
            f"pseq-{k}" : metrics.quality.pesq(x,y),
            f"stoi-{k}" : metrics.quality.stoi(x,y)
            # f"visqol-audio-{k}": metrics.quality.visqol(x, y),
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
):
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
    for i in range(len(audio_files)):
        file_results = get_metrics(audio_files[i], output / audio_files[i].name, state)
        all_results.append(file_results)
    

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
