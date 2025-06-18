#!/bin/bash
set -e
export OMP_NUM_THREADS=1

nnodes=1
nproc_per_node=8

tag=600k
dim=64
cfg=2.0
ckpt_path=/root/code/descript-audio-codec/ckpts/svae/naive_16khz_40hz_5544_kl1e-4_vae64_7khr_bigvgan_base
npy_root=/root/code/f5-svae/F5-TTS/results/naive_F5TTS_v1_Small_ldm64_40hz_lr1e-4_bsz51200_kl1e-4_ll_sm_
for ckpt in 600000 500000;do
  npy_path=${npy_root}${ckpt}
  gen_wavs=${npy_path}/ls_pc_test_clean/seed0_euler_nfe32_latent_${dim}_ss-1_cfg${cfg}_speed1.0_gt-dur
  torchrun \
  --nnodes $nnodes \
  --nproc_per_node $nproc_per_node \
  scripts/recon_latent_to_wave.py \
  --path $ckpt_path \
  --input $gen_wavs \
  --model_tag $tag
done


tag=600k
dim=64
cfg=2.0
ckpt_path=/root/code/descript-audio-codec/ckpts/svae/naive_16khz_40hz_5544_kl1e-3_vae64_7khr_bigvgan_base
npy_root=/root/code/f5-svae/F5-TTS/results/naive_F5TTS_v1_Small_ldm64_40hz_lr1e-4_bsz51200_kl1e-3_ll_sm_
for ckpt in 600000 500000;do
  npy_path=${npy_root}${ckpt}
  gen_wavs=${npy_path}/ls_pc_test_clean/seed0_euler_nfe32_latent_${dim}_ss-1_cfg${cfg}_speed1.0_gt-dur
  torchrun \
  --nnodes $nnodes \
  --nproc_per_node $nproc_per_node \
  scripts/recon_latent_to_wave.py \
  --path $ckpt_path \
  --input $gen_wavs \
  --model_tag $tag
done


tag=600k
dim=64
cfg=2.0
ckpt_path=/root/code/descript-audio-codec/ckpts/svae/16khz_40hz_5544_kl1e-2_vae64_7khr_bigvgan_base_ll_sm_attn_proj_hubert23_align_vae_no_post
npy_root=/root/code/f5-svae/F5-TTS/results/F5TTS_v1_Small_ldm64_40hz_lr1e-4_bsz51200_ll_sm_attn_proj_align_vae_hubert23_no_post_
for ckpt in 600000 500000;do
  npy_path=${npy_root}${ckpt}
  gen_wavs=${npy_path}/ls_pc_test_clean/seed0_euler_nfe32_latent_${dim}_ss-1_cfg${cfg}_speed1.0_gt-dur
  torchrun \
  --nnodes $nnodes \
  --nproc_per_node $nproc_per_node \
  scripts/recon_latent_to_wave.py \
  --path $ckpt_path \
  --input $gen_wavs \
  --model_tag $tag
done