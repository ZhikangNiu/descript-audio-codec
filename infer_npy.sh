npy_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/results/F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_latent_50hz_bzs102400_200000/ls_pc_test_clean/seed0_euler_nfe32_latent_ss-1_cfg2.0_speed1.0
npy_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/descript-audio-codec/LibriSpeech/test-clean/30hz_feat
npy_root=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/results/F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_latent_30hz_bzs102400_msk0.4-0.7_
npy_root=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/results/F5TTS_Small_vocos_char_LibriTTS_100_360_500_latent_30hz_bzs102400_msk0.4-0.7_convlayer0_
npy_root=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/results/F5TTS_Small_vocos_char_LibriTTS_100_360_500_latent_30hz_bzs102400_msk0.4-0.7_convlayer4_lr7e-5_
npy_root=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/results/F5TTS_Small_vocos_char_LibriTTS_100_360_500_latent_30hz_bzs102400_msk0.4-1.0_convlayer4_lr7e-5_
npy_root=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/results/F5TTS_Small_vocos_char_LibriTTS_100_360_500_latent_30hz_bzs51200_msk0.7-1.0_convlayer4_lr7e-5_
for ckpt in 1100000 1000000 900000 800000 700000 600000;do
    npy_path=${npy_root}${ckpt}
    echo $npy_path
# npy_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/descript-audio-codec/LibriSpeech/test-clean/30hz_feat/61/70968
# npy_path=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/F5-TTS/results/F5TTS_Small_vocos_char_LibriTTS_100_360_500_38400_latent_50hz_bzs102400_150000
    python scripts/recon_wave.py --path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/descript-audio-codec/runs/2gpu/24khz_800x_8544_kl5e-5_vae128_clamp_logvar --input=$npy_path
# python scripts/recon_wave.py --path /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/niuzhikang-240108120093/descript-audio-codec/runs/2gpu/24khz_480x_kl5e-5_vae128_clamp_logvar --input=$npy_path --model_tag 400k
done