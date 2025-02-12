import os
import string
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import multiprocessing as mp
import numpy as np
import json
import torchaudio
import torch.nn.functional as F
from ecapa_tdnn import ECAPA_TDNN_SMALL
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lang", type=str, default="en")
    parser.add_argument("-t", "--task", type=str, default="wer")
    parser.add_argument("-g", "--gen_wav_dir", type=str, required=True)
    parser.add_argument("-p", "--librispeech_test_clean_path", type=str, required=True)
    parser.add_argument("-n", "--gpu_nums", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--local", action="store_true", help="Use local custom checkpoint directory")
    return parser.parse_args()

def get_librispeech_test(gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth=False):
    """
    1) 从 gen_wav_dir 下遍历所有 .wav 文件；
    2) 获取文件名（如 174290-75945-0023.wav），进而解析出 speaker、chapter、utt_id；
    3) 在 librispeech_test_clean_path 找到对应的参考 .flac 文件和转录文本；
    4) 组装好 (gen_wav, ref_wav, ref_text) 三元组，用于后续 ASR 推理以及计算 WER。
    """

    # 获取当前目录下所有 .wav 文件的路径
    lines = list(Path(gen_wav_dir).rglob("*.flac"))
    test_set_ = []

    for wav_file in tqdm(lines):
        # wav_file 是 Path 对象，例如 /xx/xx/174290-75945-0023.wav
        # 解析出不带后缀的 stem，如 174290-75945-0023
        gen_utt = wav_file.stem
        
        # 一般情况下，LibriSpeech 的音频文件名形如：speaker-chapter-uttId
        # 例如：174290-75945-0023
        gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")

        # 在原始 Librispeech 目录下寻找参考 flac，比如：
        # librispeech_test_clean_path/speaker/chapter/speaker-chapter-uttId.flac
        ref_wav = os.path.join(
            librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, f"{gen_utt}.flac"
        )

        # 从对应的 trans.txt 中读取参考文本
        # librispeech_test_clean_path/speaker/chapter/speaker-chapter.trans.txt
        # 文件示例：
        # 174290-75945-0023 THIS IS THE GROUND TRUTH TEXT
        trans_file = os.path.join(
            librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, f"{gen_spk_id}-{gen_chaptr_id}.trans.txt"
        )
        if not os.path.exists(trans_file):
            raise FileNotFoundError(f"Transcript not found: {trans_file}")

        ref_txt = ""
        with open(trans_file, "r", encoding="utf-8") as f:
            for line_txt in f:
                line_txt = line_txt.strip()
                # 每行形如：174290-75945-0023 THIS IS THE TEXT
                if line_txt.startswith(gen_utt):
                    # 去掉前面的 utt_id，用空格分割两部分
                    ref_txt = line_txt.split(" ", 1)[1]
                    break

        if ref_txt == "":
            raise ValueError(f"No matching transcript found for {gen_utt} in {trans_file}")

        # 如果你想评估“与真实音频完全相同”时的 ASR 效果，则使用原始 flac 作为输入
        # 否则，使用你在 gen_wav_dir 下生成的合成音频 .wav 做评测
        if eval_ground_truth:
            gen_wav = ref_wav
        else:
            gen_wav = str(wav_file)
            if not os.path.exists(gen_wav):
                raise FileNotFoundError(f"Generated wav not found: {gen_wav}")

        # test_set_ 里放三元组： (合成/真实音频路径, 参考音频路径, 真值文本)
        test_set_.append((gen_wav, ref_wav, ref_txt))

    # 多卡切分
    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    splitted_test_sets = []
    for i in range(num_jobs):
        splitted_test_sets.append(
            (gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job])
        )

    return splitted_test_sets


def load_asr_model(lang, ckpt_dir=""):
    if lang == "zh":
        from funasr import AutoModel

        model = AutoModel(
            model=os.path.join(ckpt_dir, "paraformer-zh"),
            # vad_model = os.path.join(ckpt_dir, "fsmn-vad"),
            # punc_model = os.path.join(ckpt_dir, "ct-punc"),
            # spk_model = os.path.join(ckpt_dir, "cam++"),
            disable_update=True,
        )  # following seed-tts setting
    elif lang == "en":
        from faster_whisper import WhisperModel

        model_size = "large-v3" if ckpt_dir == "" else ckpt_dir
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    return model


# WER Evaluation, the way Seed-TTS does


def run_asr_wer(args):
    """
    这里和原代码一致，遍历传进来的 test_set，
    用 load_asr_model 加载模型，对输入音频做 ASR，
    然后使用 jiwer 计算 wer。
    """
    import os
    import zhconv
    from zhon.hanzi import punctuation
    import string
    from jiwer import compute_measures
    
    rank, lang, test_set, ckpt_dir = args

    # CUDA 设备绑定逻辑
    if lang == "zh":
        torch.cuda.set_device(rank)
    elif lang == "en":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    else:
        raise NotImplementedError("Only 'zh' or 'en' are supported.")

    # 加载 ASR 模型
    asr_model = load_asr_model(lang, ckpt_dir=ckpt_dir)

    punctuation_all = punctuation + string.punctuation
    wer_results = []

    for gen_wav, prompt_wav, truth in tqdm(test_set):
        # 根据语言分别调用不同模型进行推理
        if lang == "zh":
            res = asr_model.generate(input=gen_wav, batch_size_s=300, disable_pbar=True)
            hypo = res[0]["text"]
            hypo = zhconv.convert(hypo, "zh-cn")  # 繁简转换，可选
        elif lang == "en":
            segments, _ = asr_model.transcribe(gen_wav, beam_size=5, language="en")
            hypo = ""
            for segment in segments:
                hypo += " " + segment.text
        else:
            raise NotImplementedError("Only 'zh' or 'en' are supported.")

        # 计算 WER 前的文本清洗
        raw_truth = truth
        raw_hypo = hypo

        for p in punctuation_all:
            truth = truth.replace(p, "")
            hypo = hypo.replace(p, "")

        truth = truth.replace("  ", " ")
        hypo = hypo.replace("  ", " ")

        if lang == "zh":
            # 中文可以将每个汉字用空格分隔，以便 jiwer 的统计
            truth = " ".join(list(truth))
            hypo = " ".join(list(hypo))
        elif lang == "en":
            # 英文转小写
            truth = truth.lower()
            hypo = hypo.lower()

        measures = compute_measures(truth, hypo)
        wer = measures["wer"]

        wer_results.append(
            {
                "wav": Path(gen_wav).stem,
                "truth": raw_truth,
                "hypo": raw_hypo,
                "wer": wer,
            }
        )

    return wer_results

def run_sim(args):
    rank, test_set, ckpt_dir = args
    device = f"cuda:{rank}"

    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(ckpt_dir, weights_only=True, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"], strict=False)

    use_gpu = True if torch.cuda.is_available() else False
    if use_gpu:
        model = model.cuda(device)
    model.eval()

    sims = []
    for wav1, wav2, truth in tqdm(test_set):
        wav1, sr1 = torchaudio.load(wav1)
        wav2, sr2 = torchaudio.load(wav2)

        resample1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)
        resample2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)
        wav1 = resample1(wav1)
        wav2 = resample2(wav2)

        if use_gpu:
            wav1 = wav1.cuda(device)
            wav2 = wav2.cuda(device)
        with torch.no_grad():
            emb1 = model(wav1)
            emb2 = model(wav2)

        sim = F.cosine_similarity(emb1, emb2)[0].item()
        # print(f"VSim score between two audios: {sim:.4f} (-1.0, 1.0).")
        sims.append(sim)

    return sims

def main():
    args = get_args()
    lang = args.lang
    librispeech_test_clean_path = args.librispeech_test_clean_path  # test-clean path
    gen_wav_dir = args.gen_wav_dir
    eval_task = args.task
    gpus = list(range(args.gpu_nums))
    print(gpus)

    # gen_wav_dir="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/descript-audio-codec/results/2gpu/benchmark/LibriSpeech-test-clean"
    # gpus=1
    # librispeech_test_clean_path="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/speech/LibriSpeech/test-clean"
    test_set = get_librispeech_test(gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth=False)
    
    ## In LibriSpeech, some speakers utilized varying voice characteristics for different characters in the book,
    ## leading to a low similarity for the ground truth in some cases.
    # test_set = get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth = True)  # eval ground truth

    local = args.local
    if local:  # use local custom checkpoint dir
        asr_ckpt_dir = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/faster-whisper-large-v3"
    else:
        asr_ckpt_dir = ""  # auto download to cache dir
    # --------------------------- WER ---------------------------
    wavlm_ckpt_dir = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/dev_f5_be53fb1/checkpoints/wavlm_large_finetune.pth"

    if eval_task == "wer":
        wer_results = []
        wers = []

        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, lang, sub_test_set, asr_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_asr_wer, args)
            for r in results:
                wer_results.extend(r)

        wer_result_path = f"{gen_wav_dir}/{lang}_wer_results.jsonl"
        with open(wer_result_path, "w") as f:
            for line in wer_results:
                wers.append(line["wer"])
                json_line = json.dumps(line, ensure_ascii=False)
                f.write(json_line + "\n")

        wer = round(np.mean(wers) * 100, 3)
        print(f"\nTotal {len(wers)} samples")
        print(f"WER      : {wer}%")
        print(f"Results have been saved to {wer_result_path}")
        
    # --------------------------- SIM ---------------------------

    if eval_task == "sim":
        sims = []
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, sub_test_set, wavlm_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_sim, args)
            for r in results:
                sims.extend(r)

        sim = round(sum(sims) / len(sims), 3)
        print(f"\nTotal {len(sims)} samples")
        print(f"SIM      : {sim}")


if __name__ == "__main__":
    main()