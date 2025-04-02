#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
python gen_manifest.py /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriTTS/train-clean-100 --subset train-clean-100 --dest . --ext wav --seed 42
python gen_manifest.py /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriLight/vad/medium --subset ll_medium --dest . --ext flac --seed 42 
"""
import argparse
import os
import soundfile
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing audio files to index"
    )
    parser.add_argument(
        "--subset",
        default="train-clean-100"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--ext", default="flac", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included in the manifest",
    )
    return parser


def process_file(fname, path_must_contain=None):
    """
    读取单个音频文件信息，如果 `path_must_contain` 不在路径里则返回 None，
    否则返回 (绝对路径, 帧数)。
    """
    file_path = os.path.realpath(fname)
    if path_must_contain and path_must_contain not in file_path:
        return None
    frames = soundfile.info(fname).frames
    return (file_path, frames)


def main(args):
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    # 收集所有待处理的文件
    dir_path = Path(args.root)
    fnames = list(dir_path.rglob(f"**/*.{args.ext}"))
    print(f"Found {len(fnames)} files with extension: {args.ext}")

    # 多进程并行读取文件帧数
    audio_files = []
    with ProcessPoolExecutor(max_workers=16) as executor:
        # 创建提交任务
        futures = [executor.submit(process_file, f, args.path_must_contain) for f in fnames]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result is not None:
                audio_files.append(result)

    # 根据音频长度(帧数)降序排序
    audio_files.sort(key=lambda x: x[1], reverse=True)
    print(f"Valid audio files: {len(audio_files)}")

    # 写入清单文件
    out_path = os.path.join(args.dest, f"{args.subset}.lst")
    with open(out_path, "w", encoding="utf-8") as f:
        for path, frames in audio_files:
            f.write(f"{path}\t{frames}\n")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
