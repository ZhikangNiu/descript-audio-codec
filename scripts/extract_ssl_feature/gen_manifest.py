#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
python gen_manifest.py /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriTTS/train-clean-100 --subset train-clean-100 --dest . --ext wav --seed 42
"""

import argparse
import glob
import os
import soundfile
from pathlib import Path
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing flac files to index"
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


def main(args):
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    dir_path = Path(args.root)
    fnames = list(dir_path.rglob(f"**/*.{args.ext}"))
    print(f"fnames: {len(fnames)}")

    # 收集所有音频文件及其长度
    audio_files = []
    for fname in tqdm(fnames):
        file_path = os.path.realpath(fname)
        if args.path_must_contain and args.path_must_contain not in file_path:
            continue
        frames = soundfile.info(fname).frames
        audio_files.append((file_path, frames))

    # 根据音频长度降序排序
    audio_files.sort(key=lambda x: x[1], reverse=True)

    with open(os.path.join(args.dest, f"{args.subset}.lst"), "w") as train_f:
        for path, frames in audio_files:
            print(f"{path}\t{frames}", file=train_f)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
