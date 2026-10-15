#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"

# CIFAR-10, WRN-70-16, PGD-Linf (epsilon=8/255).
exec python -u -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${#gpu_ids[@]}" \
  --master_port="${MASTER_PORT:-29542}" \
  test_new.py \
  --dataset cifar10 \
  --clf_arch wideresnet-70-16 \
  --batch_size 8 \
  --attack_method pgd \
  --n_iter 200 \
  --eot 20 \
  --num_ensemble_runs 5 \
  --def_max_timesteps 1000 \
  --def_num_denoising_steps 100 \
  --def_sampling_method ddpm \
  --att_max_timesteps 1000 \
  --att_num_denoising_steps 1 \
  --att_sampling_method ddpm \
  --forward_noise_steps 50 \
  --amplitude_cut_range 10 \
  --phase_cut_range 10 \
  --delta 0.3 \
  --use_freq_guidance \
  --use_adaptive_radius \
  --radius_min 2 \
  --radius_max 12 \
  --radius_gamma 1.0 \
  --adaptive_radius_mix 0.2 \
  --adaptive_radius_threshold 0.18 \
  --adaptive_radius_temperature 0.10 \
  --lambda_amp 1.0 \
  --lambda_phase 0.25 \
  --guidance_scale 5e-4 \
  --guidance_start_ratio 0.6 \
  --seed 1 \
  --exp cifar10_wrn70_pgd \
  "$@"
