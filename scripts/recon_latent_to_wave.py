import json
from pathlib import Path

import argbind
import numpy as np
import torch
import torch.distributed as dist
from audiotools import AudioSignal
from audiotools.core import util
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from train import DAC
import torchaudio

# Custom Dataset for audio file paths
class AudioPathDataset(Dataset):
    def __init__(self, input_dir):  # Added file_extension
        self.audio_root_dir = input_dir
        self.file_lines = list(Path(input_dir).rglob("*.npy"))
        # self.file_lines = util.find_audio(input_dir, ext=[".npy"])
        print(f"All npy: {len(self.file_lines)}")

    def __len__(self):
        return len(self.file_lines)

    def __getitem__(self, idx):
        file_path = self.file_lines[idx]
        return str(file_path)

def read_json_file(metainfo_path):
    with open(metainfo_path,"r") as f:
        data = json.load(f)
    return data

def load_state(
    save_path: str,
    tag: str = "latest",
):
    folder = f"{save_path}/{tag}"
    print(f"Resuming from {str(Path('.').absolute())}/{folder}")
    metainfo_path = Path('.').absolute() / folder / "metainfo.json"
    metainfo = read_json_file(metainfo_path)
    ckpt_path = Path(folder) / "dac" / "weights.pth"
    model_dict = torch.load(ckpt_path,map_location="cpu")
    filter_dict = {k:v for k, v in model_dict["state_dict"].items() if not k.startswith("projectors")}

    generator = DAC(**metainfo["DAC"])
    del generator.projectors
    generator.load_state_dict(filter_dict, strict=True)
    generator.eval()

    return generator


@torch.no_grad()
def process(signal, generator, **kwargs):
    data = signal.audio_data.cuda()
    if signal.sample_rate != generator.sample_rate:
        signal.resample(generator.sample_rate)
    audio_data = generator.preprocess(data,signal.sample_rate)
    _, mu, log_var, _ = generator.encode(audio_data)
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
    model_tag: str = "best",
    global_seed: int = 42
):
    # Setup DDP:
    dist.init_process_group("nccl")
    # world_size is useful for DistributedSampler, batch size calculations etc.
    world_size = dist.get_world_size()
    # Batch size per GPU. For feature extraction, often 1.
    # args.global_batch_size would be the total batch across all GPUs.
    # Here, we assume DataLoader batch_size is per-GPU.
    # assert args.global_batch_size % world_size == 0, f"Global batch size must be divisible by world size."

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    
    generator = load_state(
        save_path=path,
        tag=model_tag,
    ).to(device)
    generator.eval()
    print(f"Load generator successfully")
    dataset = AudioPathDataset(input)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(
        dataset,
        batch_size=1,  # Batch size per GPU
        shuffle=False,  # Shuffle is handled by sampler
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=False,  # Process all files
    )
    
    for file_path in tqdm(loader):
        output_audio = Path(file_path[0]).with_suffix(".wav")
        if output_audio.exists():
            continue
        latent = np.load(file_path[0])
        recon = recon_wav_from_latent(latent,generator)
        torchaudio.save(output_audio,recon.squeeze(0).cpu(),sample_rate=generator.sample_rate)

    dist.destroy_process_group()

if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        get_samples()
            
