# batch size 48 , numworker 16, 4min36
# batch size 48 , numworker 8, 4min32
# batch size 48 , numworker 8 01:44, sort audio lst

import argparse
import torch
from tqdm import tqdm
import logging
import math
import os
import sys
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, AutoModel
"""
python extract_bacth_hubert.py . train-clean-100 ./LibriTTS/train-clean-100
python extract_bacth_hubert.py . train-all ./LibriTTS/
"""

# Set up logging format and level
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_semantic_model")


def compute_output_length(input_length, conv_layers):
    """
    Compute the output length after passing through convolutional layers.

    Args:
        input_length (int): The initial length of the input sequence.
        conv_layers (list): The convolutional layers in the model.

    Returns:
        int: The final output length after all convolutional layers.
    """
    for layer in conv_layers:
        K = layer.conv.kernel_size[0]  # Kernel size
        S = layer.conv.stride[0]       # Stride
        P = layer.conv.padding[0]      # Padding
        L_out = math.floor((input_length + 2 * P - K) / S + 1)  # Compute output length
        input_length = L_out
    return input_length


class AudioDataset(Dataset):
    def __init__(self, list_file, root_dir, sample_rate=16000):
        """
        Initialize the audio dataset.

        Args:
            list_file (str): File containing the relative paths of audio files.
            root_dir (str): Root directory where audio files are located.
            sample_rate (int): Target sample rate for audio processing.
        """
        with open(list_file, "r") as f:
            self.file_paths = [line.split("\t")[0] for line in f]
        self.root_dir = root_dir
        self.sample_rate = sample_rate

    def __len__(self):
        """
        Get the number of audio files in the dataset.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Load and preprocess a single audio file.

        Args:
            idx (int): Index of the audio file in the dataset.

        Returns:
            tuple: Processed audio waveform and its relative file path.
        """
        audio_path = self.file_paths[idx]
        wav, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.squeeze(0).cpu().numpy()  # Convert to numpy array
        return wav, audio_path


def collate_fn(batch):
    """
    Custom collate function for batching audio data.

    Args:
        batch (list): List of tuples (audio waveform, file path).

    Returns:
        tuple: Batch of audio waveforms and file paths.
    """
    wavs, paths = zip(*batch)
    return list(wavs), list(paths)


def main(tsv_dir, split, feat_dir, batch_size=8):
    """
    Main function to process audio data and extract features.

    Args:
        tsv_dir (str): Directory containing the list files.
        split (str): Split name (e.g., train, test).
        feat_dir (str): Directory to save the extracted features.
        batch_size (int): Batch size for processing.
    """
    # Prepare the paths
    list_file = os.path.join(tsv_dir, f"{split}.lst")
    
    # Initialize the dataset and dataloader
    wav_dataset = AudioDataset(list_file, root_dir, sample_rate=sample_rate)
    dataloader = DataLoader(
        wav_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Process each batch
    for wavs, batch_paths in tqdm(dataloader, total=len(dataloader)):
        # (b) Use feature_extractor to process all audio in the batch
        inputs = feature_extractor(
            wavs,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,  # Enable padding
        )
        input_values = inputs["input_values"].to(semantic_model.device)  # [B, T]
        attention_mask = inputs["attention_mask"].to(semantic_model.device)  # [B, T]

        # (c) Feed into the semantic model to obtain feature embeddings
        with torch.inference_mode():
            if output_hidden_states:
                outputs = semantic_model(input_values, attention_mask=attention_mask, output_hidden_states=True)
            else:
                outputs = semantic_model(input_values, attention_mask=attention_mask, output_hidden_states=False)
            batch_embedding = outputs.last_hidden_state  # [B, T', D]

        # Save the features for each file in the batch
        for i, path in enumerate(batch_paths):
            mask = attention_mask[i]  # [T]
            valid_length = mask.sum().item()  # Compute valid length
            vaild_embed_length = compute_output_length(valid_length, semantic_model.feature_extractor.conv_layers)
            single_embedding = batch_embedding[i, :vaild_embed_length, :]  # [valid_length, D]
            name = os.path.splitext(os.path.join(*(path.split("/")[-4:])))[0]
            feat_path = Path(f"{feat_dir}/{name}.npy")
            if feat_path.exists():
                continue
            os.makedirs(os.path.dirname(feat_path), exist_ok=True)
            if os.path.exists(feat_path):
                continue
            np.save(feat_path, single_embedding.detach().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir", type=str, help="Directory containing the list files.") # tsv file, you'd better sort it, can reduce padding and save time
    parser.add_argument("split", type=str, help="Dataset split name (e.g., train, test).")
    parser.add_argument("feat_dir", type=str, help="Directory to save the extracted features.")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for inference.")

    args = parser.parse_args()

    # Load the pre-trained model and feature extractor
    root_dir = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriTTS"
    semantic_model_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/hubert-large-ll60k" # you can also select hubert
    feature_extractor = AutoFeatureExtractor.from_pretrained(semantic_model_path)
    semantic_model = AutoModel.from_pretrained(semantic_model_path).eval().cuda()
    
    # Define constants
    sample_rate = 16000
    num_workers = 8
    output_hidden_states = False

    # Run the main function
    main(args.tsv_dir, args.split, args.feat_dir, batch_size=args.batch_size)