#!/bin/bash
#SBATCH -A berzelius-2026-239
#SBATCH --gpus=1
#SBATCH -t 00:02:00
#SBATCH -J test
#SBATCH -o /proj/cloudrobotics-nest/users/x_aripa/logs/%x-%j.out

set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
export PROJ=/proj/cloudrobotics-nest/users/x_aripa
export UV_CACHE_DIR=$PROJ/.cache/uv
export UV_PROJECT_ENVIRONMENT=$PROJ/envs/cloudgripper-wm

cd ~/cloudgripper-wm
nvidia-smi
uv run --locked python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
