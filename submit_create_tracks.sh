#!/bin/bash
#SBATCH --job-name=alpha_ribo_exp_pred_linear
#SBATCH --output=logs_predict_experiment_linear_%j.out
#SBATCH --error=logs_predict_experiment_linear_%j.err
#SBATCH --partition=ga100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

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

if [ ! -s sequences_human.bed.gz ]; then
    echo "Missing sequences_human.bed.gz. Run 'bash download_weights.sh' from the repo root before submitting."
    exit 1
fi
python predict_riboseq_to_bigwig.py \
    --checkpoint ../alphagenome_riboseq_head_ag_fold0_linear_poisson_multinomial_49721442.pth \
    --regions-bed ../small_regions/test_small.bed \
    --output-prefix riboseq_pred_ag_fold0_linear_poisson_multinomial-test_small

python predict_riboseq_to_bigwig.py \
    --checkpoint ../alphagenome_rnaseq_head_ag_fold0_linear_poisson_multinomial_49721436.pth \
    --regions-bed ../small_regions/test_small.bed \
    --output-prefix rnaseq_pred_ag_fold0_linear_poisson_multinomial-test_small