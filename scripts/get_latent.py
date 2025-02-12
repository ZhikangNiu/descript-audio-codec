from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools.core import util
from train import DAC
import numpy as np
from tqdm import tqdm

from dac.compare.encodec import Encodec

Encodec = argbind.bind(Encodec)
target_sample_rate=24000

def load_state(
    save_path: str,
    tag: str = "latest",
    load_weights: bool = False,
    model_type: str = "dac",
    bandwidth: float = 24.0,
):
    kwargs = {
        "folder": f"{save_path}/{tag}",
        "map_location": "cpu",
        "package": False, # NOTE: 不太确定这个有什么影响
    }
    print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")

    if model_type == "dac":
        generator, _ = DAC.load_from_folder(**kwargs)
    elif model_type == "encodec":
        generator = Encodec(bandwidth=bandwidth)

    return generator


@torch.no_grad()
def process(signal, generator, **kwargs):
    data = signal.audio_data.cuda()
    sr = signal.sample_rate
    audio_data = generator.preprocess(data,sr)
    latent, mu, log_var, kl_loss = generator.encode(audio_data)
    pre_proj_latent = generator.reparameterize(mu,log_var)
    return pre_proj_latent.transpose(1,2).cpu().numpy()


@argbind.bind(without_prefix=True)
@torch.no_grad()
def get_samples(
    path: str = "ckpt",
    input: str = "samples/input",
    output: str = "samples/output",
    model_type: str = "dac",
    model_tag: str = "best",
    bandwidth: float = 24.0,
):
    generator = load_state(
        save_path=path,
        model_type=model_type,
        bandwidth=bandwidth,
        tag=model_tag,
    ).cuda()
    generator.eval()

    audio_files = util.find_audio(input)
    print(f"Audio Nums = {len(audio_files)}")

    # global process
    # process = tracker.track("process", len(audio_files))(process)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(audio_files))):
        relative_path = audio_files[i].relative_to(input)
        output_path = output / relative_path
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        if output_path.with_suffix(".npy").exists():
            continue
        signal = AudioSignal(audio_files[i])
        if signal.sample_rate != target_sample_rate:
            signal = signal.resample(target_sample_rate)
        feat = process(signal, generator)
        
        np.save(output_path.with_suffix(".npy"),feat)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        get_samples()
            
