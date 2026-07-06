#!/bin/bash
#SBATCH --job-name=alpha_ribo_exp_linear
#SBATCH --output=logs_experiment_linear_%j.out
#SBATCH --error=logs_experiment_linear_%j.err
#SBATCH --partition=ga100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

set -eo pipefail

module load CUDA/12.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate alphagenome_ribo312
set -u

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
python -c "import torch, sys; print('python:', sys.executable); print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('torch cuda:', torch.version.cuda); x = torch.ones(1, device='cuda'); print('cuda tensor ok:', x.item()); print('gpu name:', torch.cuda.get_device_name(0))"

python train_riboseq.py \
    --checkpoint alphagenome_riboseq_head_ag_fold0_linear_poisson_multinomial.pth \
    --trunk-checkpoint models/model_fold_0.safetensors \
    --alphagenome-model-version fold_0 \
    --train-bed regions/ag_fold0/train.bed \
    --valid-bed regions/ag_fold0/valid.bed \
    --epochs 50 \
    --head-architecture linear \
    --loss-mode poisson_multinomial \
    --multinomial-segment-bp 2048 \
    --positional-weight 5.0 \
    --count-weight 1.0 \
    --min-delta 1e-5
