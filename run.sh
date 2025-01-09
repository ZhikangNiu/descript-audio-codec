export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
# torchrun --nproc_per_node $PET_NPROC_PER_NODE scripts/train.py --args.load conf/vae/24khz.yml --save_path runs/2gpu/320x_vae_dim32_grad10/
# torchrun --nproc_per_node $PET_NPROC_PER_NODE scripts/train.py --args.load conf/vae/24khz_320x_kl5e-6.yml --save_path runs/2gpu/320x_vae_dim32_grad10_kl5e-6/
# torchrun --nproc_per_node $PET_NPROC_PER_NODE scripts/train.py --args.load conf/vae/24khz_1600x_kl5e-5.yml --save_path runs/2gpu/1600x_vae_dim32_grad10_kl5e-5/
# torchrun --nproc_per_node $PET_NPROC_PER_NODE scripts/train.py --args.load conf/vae/24khz_800x_8544_kl5e-5.yml --save_path runs/2gpu/800x_8544_vae_dim32_grad10_kl5e-5/
# torchrun --nproc_per_node 2 scripts/train.py --args.load conf/vae/24khz_1600x_kl5e-5_vae128.yml --save_path runs/2gpu/24khz_1600x_kl5e-5_vae128/
# torchrun --nproc_per_node 2 scripts/train.py --args.load conf/vae/24khz_1600x_kl5e-4_vae128.yml --save_path runs/2gpu/24khz_1600x_kl5e-4_vae128/ 
torchrun --nproc_per_node 2 scripts/train.py --args.load conf/vae/24khz_1600x_kl5e-3_vae128.yml --save_path runs/2gpu/24khz_1600x_kl5e-3_vae128