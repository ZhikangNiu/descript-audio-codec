import argparse
import torch
from tqdm import tqdm
import logging
import math
import os
import sys
import numpy as np
import torchaudio
from torch import distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler, SequentialSampler
from transformers import AutoFeatureExtractor, AutoModel

"""
Example usage:
    # Single GPU
    python extract_batch_hubert.py --tsv_dir . --split train-clean-100 --feat_dir ./LibriTTS/train-clean-100

    # Multi-GPU (4 GPUs):
    export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=8 extract_hubert_batch_multi_gpu.py ./libritts_metalst lt_100 ./hubert_large_66k_last_layer/LibriTTS --batch_size 80 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/hubert-large-ll60k
   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=8 extract_hubert_batch_multi_gpu.py ./libriheavy_metalst lh_medium ./wavlm_large_last_layer/libriheavy --batch_size 80 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/wavlm-large 
   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=4 extract_hubert_batch_multi_gpu.py ./libritts_metalst lt_100 ./wavlm_large_last_layer/LibriTTS --batch_size 48 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/wavlm-large 
   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=4 extract_hubert_batch_multi_gpu.py ./libritts_metalst lt_100 ./wavlm_large_9_layer/LibriTTS --batch_size 48 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/wavlm-large  --output_hidden_state --layer 9
   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=4 extract_hubert_batch_multi_gpu.py ./libritts_metalst lt_500 ./wavlm_large_9_layer/LibriTTS --batch_size 48 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/wavlm-large  --output_hidden_state --layer 9
   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=4 extract_hubert_batch_multi_gpu.py ./libriheavy_metalst lh_medium ./wavlm_large_9_layer/libriheavy --batch_size 48 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/wavlm-large  --output_hidden_state --layer 9
   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=4 extract_hubert_batch_multi_gpu.py ./libritts_metalst lt_100 ./wavlm_large_23_layer/LibriTTS --batch_size 48 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/wavlm-large  --output_hidden_state --layer 23
   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=5 extract_hubert_batch_multi_gpu.py ./libritts_metalst lt_100 ./hubert_large_avg_layer/LibriTTS --batch_size 48 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/hubert-large-ll60k  --output_hidden_state # extract avg feature
   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=5 extract_hubert_batch_multi_gpu.py ./libritts_metalst lt_360 ./hubert_large_avg_layer/LibriTTS --batch_size 32 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/hubert-large-ll60k  --output_hidden_state
   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=5 extract_hubert_batch_multi_gpu.py ./libritts_metalst lt_500 ./hubert_large_avg_layer/LibriTTS --batch_size 32 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/hubert-large-ll60k  --output_hidden_state

   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=8 extract_hubert_batch_multi_gpu.py ./libritts_metalst lt_100 ./wavlm_large_avg_layer/LibriTTS --batch_size 32 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/wavlm-large  --output_hidden_state
   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=8 extract_hubert_batch_multi_gpu.py ./libritts_metalst lt_360 ./wavlm_large_avg_layer/LibriTTS --batch_size 32 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/wavlm-large  --output_hidden_state
   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=8 extract_hubert_batch_multi_gpu.py ./libritts_metalst lt_500 ./wavlm_large_avg_layer/LibriTTS --batch_size 32 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/wavlm-large  --output_hidden_state
   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=8 extract_hubert_batch_multi_gpu.py ./libriheavy_metalst lh_medium ./wavlm_large_avg_layer/LibriTTS --batch_size 32 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/wavlm-large  --output_hidden_state
   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=8 extract_hubert_batch_multi_gpu.py ./libriheavy_metalst lh_small ./wavlm_large_avg_layer/LibriTTS --batch_size 32 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/wavlm-large  --output_hidden_state

   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=8 extract_hubert_batch_multi_gpu.py ./libritts_metalst lt_500 ./wavlm_large_23_layer/LibriTTS --batch_size 24 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/wavlm-large  --output_hidden_state --layer 23
   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=8 extract_hubert_batch_multi_gpu.py ./libritts_metalst lt_100 ./wavlm_large_23_layer/LibriTTS --batch_size 24 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/wavlm-large  --output_hidden_state --layer 23
   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=8 extract_hubert_batch_multi_gpu.py ./libritts_metalst lt_360 ./wavlm_large_23_layer/LibriTTS --batch_size 24 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/wavlm-large  --output_hidden_state --layer 23
   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=8 extract_hubert_batch_multi_gpu.py ./libriheavy_metalst lh_medium ./wavlm_large_23_layer/libriheavy --batch_size 24 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/wavlm-large  --output_hidden_state --layer 23
   export OMP_NUM_THREADS=1 && torchrun --nproc_per_node=8 extract_hubert_batch_multi_gpu.py ./libriheavy_metalst lh_small ./wavlm_large_23_layer/libriheavy --batch_size 24 --semantic_model_path /vepfs/user/tts/sii_F5TTS/checkpoints/wavlm-large  --output_hidden_state --layer 23
"""

# Set up logging format and level
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_semantic_model")

ERROR_LOG_FILE = "error_audiofiles.txt"

def log_error(audio_path, error_msg):
    """Write erroneous audiofile paths and messages into a log file."""
    with open(ERROR_LOG_FILE, "a") as f:
        f.write(f"{audio_path}\t{error_msg}\n")

def compute_output_length(input_length, conv_layers):
    """
    Compute the output length after passing through convolutional layers.
    Typically used for wav2vec2 / HuBERT feature extractors.
    """
    for layer in conv_layers:
        K = layer.conv.kernel_size[0]  # Kernel size
        S = layer.conv.stride[0]       # Stride
        P = layer.conv.padding[0]      # Padding
        L_out = math.floor((input_length + 2 * P - K) / S + 1)
        input_length = L_out
    return input_length


class AudioDataset(Dataset):
    def __init__(self, list_file, feat_dir, sample_rate=16000):
        """
        Initialize the audio dataset.
        Skip any audio for which the .npy feature file already exists.
        """
        valid_paths = []
        with open(list_file, "r") as f:
            for line in f:
                audio_path = line.strip().split("\t")[0]
                # Derive the final feature file path
                name = os.path.splitext(os.path.join(*(audio_path.split("/")[-4:])))[0]
                feat_path = os.path.join(feat_dir, f"{name}.npy")
                # Skip if feature file already exists
                if os.path.exists(feat_path):
                    logger.info(f"{feat_path} has already been processed; skipping.")
                    continue
                valid_paths.append(audio_path)

        self.file_paths = valid_paths
        self.feat_dir = feat_dir
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        try:
            wav, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            wav = wav.squeeze(0).cpu().numpy()  # Convert to numpy array
            return wav, audio_path
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing {audio_path}: {error_msg}")
            log_error(audio_path, error_msg)
            return None


def collate_fn(batch):
    """
    Custom collate function for batching audio data.
    Filter out any samples that failed to load (None).
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return [], []
    wavs, paths = zip(*batch)
    return list(wavs), list(paths)


def run_feature_extraction(
    rank,
    world_size,
    args,
    feature_extractor,
    semantic_model,
    sample_rate=16000,
    num_workers=8,
    output_hidden_states=False,
    layer=-1
):
    """
    Main logic for creating DataLoader, distributing data if needed, and extracting features.
    """

    # 1. Build the dataset
    list_file = os.path.join(args.tsv_dir, f"{args.split}.lst")
    dataset = AudioDataset(list_file, args.feat_dir, sample_rate=sample_rate)
    logger.info(f"Rank {rank}: Dataset size after skipping existing .npy = {len(dataset)}")

    # 2. Build the sampler
    if world_size > 1:
        # For multi-GPU, use DistributedSampler to split the dataset across ranks
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,    # No shuffle so that order is consistent
            drop_last=False,
        )
    else:
        # Single GPU (or CPU) – no distributed sampling
        sampler = SequentialSampler(dataset)

    # 3. Build the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    semantic_model = semantic_model.to(device)

    # 4. Loop over DataLoader
    for wavs, batch_paths in tqdm(dataloader, total=len(dataloader), desc=f"Rank {rank}"):
        if len(wavs) == 0:
            continue

        # (a) Use feature_extractor to process all audio in the batch
        inputs = feature_extractor(
            wavs,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,  # Enable padding
        )
        input_values = inputs["input_values"].to(device)  # [B, T]
        attention_mask = inputs["attention_mask"].to(device)  # [B, T]

        # (b) Feed into the semantic model
        with torch.inference_mode():
            if output_hidden_states:
                outputs = semantic_model(
                        input_values,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                hidden_states = outputs.hidden_states
                if layer > -1:
                    batch_embedding = hidden_states[layer]
                elif layer == -1:
                    hidden_states = torch.stack(hidden_states, dim=0)
                    batch_embedding = hidden_states.mean(dim=0)
            else:
                outputs = semantic_model(
                    input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=False
                )
                batch_embedding = outputs.last_hidden_state  # [B, T', D]

        # (c) Save features for each file in the batch
        for i, path in enumerate(batch_paths):
            try:
                mask = attention_mask[i]  # [T]
                valid_length = mask.sum().item()  # Compute valid length
                valid_embed_length = compute_output_length(
                    valid_length,
                    semantic_model.feature_extractor.conv_layers
                )
                # print(batch_embedding.shape)
                single_embedding = batch_embedding[i, :valid_embed_length, :]  # [valid_length, D]

                name = os.path.splitext(os.path.join(*(path.split("/")[-4:])))[0]
                feat_path = os.path.join(args.feat_dir, f"{name}.npy")
                os.makedirs(os.path.dirname(feat_path), exist_ok=True)
                if os.path.exists(feat_path):
                    # In case of concurrency or re-check
                    continue
                np.save(feat_path, single_embedding.detach().cpu().numpy())
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error processing batch file {path}: {error_msg}")
                log_error(path, error_msg)
                continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir", type=str, help="Directory containing the list files.")
    parser.add_argument("split", type=str, help="Dataset split name (e.g., train, test).")
    parser.add_argument("feat_dir", type=str, help="Directory to save the extracted features.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--semantic_model_path", type=str, default="/vepfs/user/tts/sii_F5TTS/checkpoints/hubert-large-ll60k",
                        help="Path or HuggingFace Hub name for semantic model.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers.")
    parser.add_argument("--output_hidden_states", action="store_true",
                        help="If set, model will return all hidden states.")
    parser.add_argument("--layer",default=-1,type=int)
    args = parser.parse_args()

    # Distributed Setup
    # If WORLD_SIZE > 1, we assume multi-GPU distributed mode.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        # Initialize process group
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Load model + feature extractor (only on rank=0 or everyone if you prefer; 
    # for inference, each rank needs a copy)
    if rank == 0:
        logger.info(f"Loading model from {args.semantic_model_path}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.semantic_model_path)
    semantic_model = AutoModel.from_pretrained(args.semantic_model_path).eval()

    # Run the feature-extraction process
    run_feature_extraction(
        rank=rank,
        world_size=world_size,
        args=args,
        feature_extractor=feature_extractor,
        semantic_model=semantic_model,
        sample_rate=16000,
        num_workers=args.num_workers,
        output_hidden_states=args.output_hidden_states
    )

    # Finalize
    if world_size > 1:
        dist.destroy_process_group()
    if rank == 0:
        logger.info("Feature extraction complete.")


if __name__ == "__main__":
    main()
