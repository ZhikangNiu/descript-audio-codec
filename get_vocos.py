import torchaudio
from pathlib import Path
import numpy as np
from tqdm import tqdm
def get_vocos_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(waveform.shape) == 2

    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel

libritts_test_clean = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriTTS/train-clean-100/"
output_path = "./LibriTTS/vocos/train-clean-100"
file_list = list(Path(libritts_test_clean).rglob("*.wav"))
for file in tqdm(file_list):
    wav,sr = torchaudio.load(file)
    mel = get_vocos_mel_spectrogram(wav.unsqueeze(0))
    relative_path = file.relative_to(libritts_test_clean)
    output = Path(output_path) / relative_path
    if not Path(output).parent.exists():
        Path(output).parent.mkdir(parents=True)
    
    np.save(output,mel.squeeze(0).cpu().numpy())
    # import ipdb;ipdb.set_trace()
    