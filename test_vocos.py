import torch
from vocos import Vocos
from pathlib import Path
import glob
import torchaudio
import argparse
import os



lt_root="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriTTS"
vocos_ckpt = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/vocos-mel-24khz"
bigvgan_ckpt = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/bigvgan_v2_24khz_100band_256x/"
stable_codec_ckpt = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/stable-codec/ckpt/stable-codec-speech-16k"

ckpt_dict = {
    "vocos" : vocos_ckpt,
    "bigvgan": bigvgan_ckpt,
    "stable-codec": stable_codec_ckpt
}
AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".mp4"]

def find_audio(folder: str, ext=AUDIO_EXTENSIONS):
    folder = Path(folder)
    # Take care of case where user has passed in an audio file directly
    # into one of the calling functions.
    if str(folder).endswith(tuple(ext)):
        # if, however, there's a glob in the path, we need to
        # return the glob, not the file.
        if "*" in str(folder):
            return glob.glob(str(folder), recursive=("**" in str(folder)))
        else:
            return [folder]

    files = []
    for x in ext:
        files += folder.glob(f"**/*{x}")
    return files

def load_vocoder(vocoder_name="vocos", is_local=False, local_path="", device="cuda", hf_cache_dir=None):
    if vocoder_name == "vocos":
        # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        if is_local:
            print(f"Load vocos from local path {local_path}")
            config_path = f"{local_path}/config.yaml"
            model_path = f"{local_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")
        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "bigvgan":
        if is_local:
            """download from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main"""
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)
        else:
            local_path = snapshot_download(repo_id="nvidia/bigvgan_v2_24khz_100band_256x", cache_dir=hf_cache_dir)
            vocoder = bigvgan.BigVGAN.from_pretrained(local_path, use_cuda_kernel=False)

        vocoder.remove_weight_norm()
        vocoder = vocoder.eval().to(device)
    elif vocoder_name == "stable-codec":
        vocoder = StableCodec(
            model_config_path=os.path.join(local_path,"model_config.json"),
            ckpt_path=os.path.join(local_path,"model.safetensors"), # optional, can be `None`,
            device = torch.device("cuda") # not support cpu 
        )
    return vocoder

@torch.inference_mode
def cli_main(
    input: str = "samples/input",
    output: str = "samples/output",
    subset: str = "test-other",
    vocoder_name: str = "vocos",
    
):
    input_subset = Path(input)/subset
    audio_files = find_audio(input_subset)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    
    local_ckpt_path = ckpt_dict[vocoder_name]
    vocoder = load_vocoder(
        vocoder_name=vocoder_name,
        is_local=True,
        local_path=local_ckpt_path,
        device="cuda"
    )
    
    for file in audio_files:
        if vocoder_name in ["vocos","bigvgan"]:
            y, sr = torchaudio.load(file)
            length = y.size(1)
            if y.size(0) > 1:  # mix to mono
                y = y.mean(dim=0, keepdim=True)
            if sr != 24000:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)
            if vocoder_name == "vocos":
                y_hat = vocoder(y.cuda())[:,:length].cpu()
            else:
                mel = get_mel_spectrogram(y, vocoder.h).cuda()
                y_hat = vocoder(mel).squeeze(0).cpu()
            torchaudio.save(output / file.name, y_hat, 24000)
        elif vocoder_name == "stable-codec":
            try:
                latents, tokens = vocoder.encode(str(file))
                decoded_audio = vocoder.decode(tokens)
                torchaudio.save(output / file.name, decoded_audio.squeeze(0).cpu(), vocoder.sample_rate)
            except:
                print(file)

@torch.inference_mode
def check_one_waveform(
    input: str = "61/",
    output: str = "./61_recon/",
    vocoder_name: str = "vocos",
    
):
    audio_files = Path(input).rglob("**/*.flac")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    
    local_ckpt_path = ckpt_dict[vocoder_name]
    vocoder = load_vocoder(
        vocoder_name=vocoder_name,
        is_local=True,
        local_path=local_ckpt_path,
        device="cuda"
    )
    
    for file in audio_files:
        if vocoder_name in ["vocos","bigvgan"]:
            y, sr = torchaudio.load(file)
            
            if y.size(0) > 1:  # mix to mono
                y = y.mean(dim=0, keepdim=True)
            if sr != 24000:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)
                length = y.size(1)
            if vocoder_name == "vocos":
                # mel = vocoder.feature_extractor(y.cuda())
                # import ipdb;ipdb.set_trace()
                y_hat = vocoder(y.cuda()).cpu()
            else:
                mel = get_mel_spectrogram(y, vocoder.h).cuda()
                y_hat = vocoder(mel).squeeze(0).cpu()
            torchaudio.save(output / file.name, y_hat, 24000)
        elif vocoder_name == "stable-codec":
            try:
                latents, tokens = vocoder.encode(str(file))
                decoded_audio = vocoder.decode(tokens)
                torchaudio.save(output / file.name, decoded_audio.squeeze(0).cpu(), vocoder.sample_rate)
            except:
                print(file)

        
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Vocoder Inference Script")
    # parser.add_argument("--input", type=str, default=lt_root, help="Input folder or file containing audio data")
    # parser.add_argument("--output", type=str, required=True, help="Output folder to save the processed audio")
    # parser.add_argument("--subset", type=str, default="test-other", help="Subset folder name (default: test-other)")
    # parser.add_argument("--vocoder_name", type=str, default="vocos", help="Name of the vocoder (default: vocos)")

    # args = parser.parse_args()
    
    # cli_main(
    #     input=args.input,
    #     output=args.output,
    #     subset=args.subset,
    #     vocoder_name=args.vocoder_name
    # )
    check_one_waveform()

    # cli_main(input=lt_root,output="results/2gpu/benchmark/vocos",subset="test-other")