cat > compare_rnaseq_riboseq.py << 'EOF'
import pyBigWig
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import gzip
import re
import os

# ── Paths ──────────────────────────────────────────────────────────────────
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
GTF_PATH = "/nemo/project/proj-ai-dna-hackathon/proj5/ref/gencode.v44.primary_assembly.annotation.longest_cds_transcripts.gtf.gz"

bigwigs = {
    'rnaseq_plus':   f"{WORK_DIR}/rnaseq_pred_ag_fold0_linear_poisson_multinomial-test_small.plus.bigwig",
    'rnaseq_minus':  f"{WORK_DIR}/rnaseq_pred_ag_fold0_linear_poisson_multinomial-test_small.minus.bigwig",
    'riboseq_plus':  f"{WORK_DIR}/riboseq_pred_ag_fold0_linear_poisson_multinomial-test_small.plus.bigwig",
    'riboseq_minus': f"{WORK_DIR}/riboseq_pred_ag_fold0_linear_poisson_multinomial-test_small.minus.bigwig",
}

# ── Check files exist ──────────────────────────────────────────────────────
print("🔍 Checking files...")
for name, path in bigwigs.items():
    exists = os.path.exists(path)
    print(f"  {'✅' if exists else '❌'} {name}: {path}")
print(f"  {'✅' if os.path.exists(GTF_PATH) else '❌'} GTF: {GTF_PATH}")

# ── Get chromosomes available in BigWigs ───────────────────────────────────
print("\n🔍 Checking available chromosomes in BigWigs...")
with pyBigWig.open(bigwigs['rnaseq_plus']) as bw:
    available_chroms = set(bw.chroms().keys())
    print(f"  Available chroms: {sorted(available_chroms)[:10]}...")

# ── Parse GTF for CDS regions ──────────────────────────────────────────────
def parse_gtf_cds(gtf_path, valid_chroms, max_regions=1000):
    """Extract CDS intervals from GTF file"""
    cds_regions = []
    opener = gzip.open if gtf_path.endswith('.gz') else open

    with opener(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            if fields[2] != 'CDS':
                continue

            chrom  = fields[0]
            start  = int(fields[3]) - 1  # GTF is 1-based → 0-based
            end    = int(fields[4])
            strand = fields[6]

            # Only chromosomes present in BigWigs
            if chrom not in valid_chroms:
                continue
            # Skip very short CDS
            if end - start < 100:
                continue

            cds_regions.append({
                'chrom':  chrom,
                'start':  start,
                'end':    end,
                'strand': strand
            })

            if len(cds_regions) >= max_regions:
                break

    return pd.DataFrame(cds_regions)

print("\n📖 Parsing GTF for CDS regions...")
cds_df = parse_gtf_cds(GTF_PATH, available_chroms, max_regions=1000)
print(f"✅ Found {len(cds_df)} CDS regions")
print(cds_df.head())
print(f"\nChromosome distribution:\n{cds_df['chrom'].value_counts().head(10)}")

# ── Extract BigWig signal over CDS regions ─────────────────────────────────
def get_mean_signal(bw_plus, bw_minus, chrom, start, end, strand):
    """Get mean signal over a region from the correct strand"""
    try:
        bw = bw_plus if strand == '+' else bw_minus
        vals = bw.values(chrom, start, end, numpy=True)
        vals = np.nan_to_num(vals, nan=0.0)
        return float(vals.mean())
    except Exception as e:
        return np.nan

print("\n📊 Extracting signals over CDS regions...")
results = []

with pyBigWig.open(bigwigs['rnaseq_plus'])  as rna_plus,  \
     pyBigWig.open(bigwigs['rnaseq_minus']) as rna_minus, \
     pyBigWig.open(bigwigs['riboseq_plus'])  as ribo_plus, \
     pyBigWig.open(bigwigs['riboseq_minus']) as ribo_minus:

    for i, row in cds_df.iterrows():
        rna_sig  = get_mean_signal(rna_plus,  rna_minus,
                                   row['chrom'], row['start'],
                                   row['end'],   row['strand'])
        ribo_sig = get_mean_signal(ribo_plus, ribo_minus,
                                   row['chrom'], row['start'],
                                   row['end'],   row['strand'])
        results.append({
            'chrom':   row['chrom'],
            'start':   row['start'],
            'end':     row['end'],
            'strand':  row['strand'],
            'rnaseq':  rna_sig,
            'riboseq': ribo_sig
        })

        if i % 100 == 0:
            print(f"  Processed {i}/{len(cds_df)} regions...")

results_df = pd.DataFrame(results).dropna()
# Remove regions with zero signal in both
results_df = results_df[
    (results_df['rnaseq'] > 0) | (results_df['riboseq'] > 0)
]
print(f"✅ Valid regions with signal: {len(results_df)}")

# ── Compute Correlations ───────────────────────────────────────────────────
pearson_r,  pearson_p  = pearsonr( results_df['rnaseq'], results_df['riboseq'])
spearman_r, spearman_p = spearmanr(results_df['rnaseq'], results_df['riboseq'])

print(f"\n📈 Correlation Results (over CDS regions):")
print(f"   Pearson  r = {pearson_r:.4f}  (p = {pearson_p:.2e})")
print(f"   Spearman r = {spearman_r:.4f}  (p = {spearman_p:.2e})")
print(f"   N regions  = {len(results_df)}")

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Linear scale
axes[0].scatter(results_df['rnaseq'], results_df['riboseq'],
                alpha=0.4, s=15, color='steelblue', edgecolors='none')
axes[0].set_xlabel('RNA-seq predicted signal (mean over CDS)', fontsize=11)
axes[0].set_ylabel('Ribo-seq predicted signal (mean over CDS)', fontsize=11)
axes[0].set_title(f'RNA-seq vs Ribo-seq Predictions over CDS\n'
                  f'Pearson r={pearson_r:.3f}  Spearman r={spearman_r:.3f}  N={len(results_df)}',
                  fontsize=11)

# Log scale
axes[1].scatter(np.log1p(results_df['rnaseq']),
                np.log1p(results_df['riboseq']),
                alpha=0.4, s=15, color='coral', edgecolors='none')
axes[1].set_xlabel('log1p(RNA-seq predicted signal)', fontsize=11)
axes[1].set_ylabel('log1p(Ribo-seq predicted signal)', fontsize=11)
axes[1].set_title(f'RNA-seq vs Ribo-seq Predictions over CDS (log scale)\n'
                  f'Pearson r={pearson_r:.3f}  Spearman r={spearman_r:.3f}  N={len(results_df)}',
                  fontsize=11)

plt.tight_layout()
plt.savefig('rnaseq_vs_riboseq_correlation.png', dpi=150, bbox_inches='tight')
print("✅ Plot saved: rnaseq_vs_riboseq_correlation.png")

# Save results table
results_df.to_csv('rnaseq_vs_riboseq_cds_signals.csv', index=False)
print("✅ Results saved: rnaseq_vs_riboseq_cds_signals.csv")
EOF