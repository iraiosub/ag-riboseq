# ag-riboseq

Ribo-seq resources for the AlphaGenome hackathon project. The lightweight files
in this repository document the environment and the expected project layout; the
large data and model artifacts live on Nemo.

Project path on Nemo:

`/nemo/project/proj-ai-dna-hackathon/proj5`

## Environment

The conda environment is defined in `env.yml`:

```bash
conda env create -f env.yml
conda activate alphagenome_ribo312
```

The environment includes PyTorch, CUDA support, `safetensors`, `pyBigWig`, and
`pyfaidx`, which are the key dependencies for loading the model weights and
reading the genomic signal/reference files.

## Data and Model Layout

Expected contents under the Nemo project directory:

```
.
├── bigwig
│   ├── human_brain_ribo_1.psites.forward.bigWig
│   ├── human_brain_ribo_1.psites.forward.rpm.bigWig
│   ├── human_brain_ribo_1.psites.reverse.bigWig
│   ├── human_brain_ribo_1.psites.reverse.rpm.bigWig
│   ├── human_brain_ribo_2.psites.forward.bigWig
│   ├── human_brain_ribo_2.psites.forward.rpm.bigWig
│   ├── human_brain_ribo_2.psites.reverse.bigWig
│   └── human_brain_ribo_2.psites.reverse.rpm.bigWig
├── models
│   ├── model_all_folds.safetensors
│   └── model_fold_0.safetensors
├── ref
│   ├── gencode.v44.pc_translations.fa.gz
│   ├── gencode.v44.primary_assembly.annotation.gtf
│   ├── gencode.v44.primary_assembly.annotation.longest_cds_transcripts.fa
│   ├── gencode.v44.primary_assembly.annotation.longest_cds_transcripts.gtf.gz
│   ├── GRCh38.primary_assembly.genome.fa
│   ├── GRCh38.primary_assembly.genome.fa.fai
│   └── GRCh38.primary_assembly.genome.fa.gz
└── regions
    ├── ag_fold0
    │   ├── test.bed
    │   ├── train.bed
    │   └── valid.bed
    └── sequences_human.bed.gz
```

## Files

### `bigwig/`

Ribo-seq P-site signal tracks for human brain samples.

- `human_brain_ribo_1...` and `human_brain_ribo_2...` are two Ribo-seq samples
  or replicates.
- `forward` and `reverse` files store strand-specific P-site signal.
- `.rpm.bigWig` files are reads-per-million normalized versions of the same
  strand/sample tracks.
- non-`.rpm.bigWig` files keep the original unnormalized signal.

These tracks are the observed ribosome occupancy signal used for training,
evaluation, or visualization.

### `models/`

AlphaGenome model weights saved in `safetensors` format.

- `model_fold_0.safetensors` contains the checkpoint for the model trained for
  fold 0, corresponding to the fold split in `regions/ag_fold0/`.
- `model_all_folds.safetensors` contains the all-folds/final checkpoint. Use
  this for general inference when you are not specifically reproducing the
  held-out fold 0 evaluation.

### `ref/`

Reference genome and annotation files.

- `GRCh38.primary_assembly.genome.fa` is the uncompressed GRCh38 primary
  assembly FASTA.
- `GRCh38.primary_assembly.genome.fa.fai` is the FASTA index used by tools such
  as `pyfaidx`.
- `GRCh38.primary_assembly.genome.fa.gz` is the compressed copy of the same
  reference genome.
- `gencode.v44.primary_assembly.annotation.longest_cds_transcripts.gtf.gz`
  contains GENCODE v44 transcript annotations filtered to the longest CDS
  transcript per gene.

### `regions/`

BED files defining the genomic intervals used by the project.

- `sequences_human.bed.gz` is the full set of human sequence intervals.
- `ag_fold0/train.bed`, `ag_fold0/valid.bed`, and `ag_fold0/test.bed` split
  those intervals into training, validation, and held-out test sets for fold 0.
