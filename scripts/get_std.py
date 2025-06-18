import numpy as np
from pathlib import Path
import torch
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/home/ubuntu/data/f5_tts/data/train/wavs")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes. Default is CPU count.")
    return parser.parse_args()


def process_file(file_path):
    try:
        data = np.load(file_path)
        latent = torch.from_numpy(data).squeeze(0).mul(0.486)
        return latent
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def get_std(path: Path, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()
    
    # 获取所有文件路径
    file_paths = list(path.rglob("*.npy"))[:100000]
    print(f"Found {len(file_paths)} files to process")
    
    # 使用进程池处理文件
    with Pool(processes=num_workers) as pool:
        # results = pool.map(process_file, file_paths)
        results = list(
            tqdm(
                pool.imap(process_file, file_paths),
                total=len(file_paths)
            )
        )
    
    # 过滤掉处理失败的结果
    results = [r for r in results if r is not None]
    
    if not results:
        raise ValueError("No valid data was processed")
    
    # 合并所有数据
    all_data = torch.cat(results, dim=1)
    std = all_data.std().item()
    mean = all_data.mean().item()
    normalizer = 1 / std
    return normalizer, mean, std


if __name__ == "__main__":
    args = get_args()
    normalizer, mean, std = get_std(Path(args.path), args.num_workers)
    print(f"Normalizer: {normalizer}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")