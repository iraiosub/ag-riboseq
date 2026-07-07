#!/bin/bash
#SBATCH --job-name=rnaseq_ribo_corr
#SBATCH --output=logs_correlation_%j.out
#SBATCH --error=logs_correlation_%j.err
#SBATCH --partition=ga100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -eo pipefail

module load CUDA/12.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate alphagenome_ribo312
set -u

echo "Running on host: $(hostname)"
python compare_rnaseq_riboseq.py
