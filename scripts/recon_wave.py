from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools.core import util
from train import DAC
import numpy as np
from tqdm import tqdm
import torchaudio
import json
from dac.compare.encodec import Encodec

Encodec = argbind.bind(Encodec)

def read_json_file(metainfo_path):
    with open(metainfo_path,"r") as f:
        data = json.load(f)
    return data

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
    metainfo_path = Path('.').absolute()/kwargs['folder']/"metainfo.json"
    metainfo = read_json_file(metainfo_path)
    ckpt_path = Path(kwargs["folder"]) / "dac" / "weights.pth"
    model_dict = torch.load(ckpt_path,map_location=kwargs["map_location"])
    filter_dict = {k:v for k, v in model_dict["state_dict"].items() if not k.startswith("projectors")}
    if model_type == "dac":
        generator = DAC(**metainfo["DAC"])
        del generator.projectors
        generator.load_state_dict(filter_dict, strict=True)
        generator.eval()
    elif model_type == "encodec":
        generator = Encodec(bandwidth=bandwidth)

    return generator


@torch.no_grad()
def process(signal, generator, **kwargs):
    data = signal.audio_data.cuda()
    sr = signal.sample_rate
    audio_data = generator.preprocess(data,sr)
    latent, mu, log_var, kl_loss = generator.encode(data)
    pre_proj_latent = generator.reparameterize(mu,log_var)
    return pre_proj_latent.transpose(1,2).cpu().numpy()

@torch.no_grad()
def recon_wav_from_latent(latent_data,generator):
    z_hat = torch.from_numpy(latent_data).cuda()
   
    final_result = generator.decode(z_hat.transpose(1,2))
    return final_result

@argbind.bind(without_prefix=True)
@torch.no_grad()
def get_samples(
    path: str = "ckpt",
    input: str = "samples/input",
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
    audio_files = list(Path(input).rglob("*.npy"))
    print(f"Audio Nums = {len(audio_files)}")
    print(f"Audio Generator SR: {generator.sample_rate}")

    for i in tqdm(range(len(audio_files))):
        output_audio = audio_files[i].with_suffix(".wav")
        if output_audio.exists():
            continue
        latent = np.load(audio_files[i])
        recon = recon_wav_from_latent(latent,generator)
        torchaudio.save(output_audio,recon.squeeze(0).cpu(),sample_rate=generator.sample_rate)

if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        get_samples()
            
