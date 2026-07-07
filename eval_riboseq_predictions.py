import argparse
import csv
import gzip
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyBigWig
from bigwig_replicates import BigWigReplicates
from tqdm import tqdm


RIBO_BIGWIG_DIR = "/camp/lab/ulej/home/users/luscomben/users/iosubi/projects/ag/riboseq/bigwig"
OBS_PLUS = ",".join([
    f"{RIBO_BIGWIG_DIR}/human_brain_ribo_1.psites.forward.rpm.bigWig",
    f"{RIBO_BIGWIG_DIR}/human_brain_ribo_2.psites.forward.rpm.bigWig",
])
OBS_MINUS = ",".join([
    f"{RIBO_BIGWIG_DIR}/human_brain_ribo_1.psites.reverse.rpm.bigWig",
    f"{RIBO_BIGWIG_DIR}/human_brain_ribo_2.psites.reverse.rpm.bigWig",
])
PRED_PLUS = "riboseq_pred_ag_fold0_linear_poisson_multinomial.plus.bigwig"
PRED_MINUS = "riboseq_pred_ag_fold0_linear_poisson_multinomial.minus.bigwig"
REGIONS_BED = "regions/ag_fold0/test.bed"
DEFAULT_GTF = "/camp/lab/ulej/home/shared/oscar_ira_riboloco/ref/human/gencode.v44.primary_assembly.annotation.longest_cds_transcripts.gtf.gz"
START_CODON_FRAME_REGIONS = ("utr5", "cds", "utr3")
START_CODON_FRAME_SOURCES = (("obs", "observed"), ("pred", "predicted"))
CORRELATION_REGIONS = ("cds", "utr5", "utr5_distal", "utr3")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted Ribo-seq BigWigs against observed P-site BigWigs."
    )
    parser.add_argument(
        "--observed-plus",
        default=OBS_PLUS,
        help="Plus-strand observed BigWig path(s), comma-separated for replicate averaging.",
    )
    parser.add_argument(
        "--observed-minus",
        default=OBS_MINUS,
        help="Minus-strand observed BigWig path(s), comma-separated for replicate averaging.",
    )
    parser.add_argument("--predicted-plus", default=PRED_PLUS)
    parser.add_argument("--predicted-minus", default=PRED_MINUS)
    parser.add_argument("--regions-bed", default=REGIONS_BED)
    parser.add_argument(
        "--chroms",
        default=None,
        help=(
            "Optional comma-separated chromosomes to evaluate, e.g. chr2. "
            "Applied immediately after loading the regions BED."
        ),
    )
    parser.add_argument("--out-prefix", default="riboseq_eval")
    parser.add_argument("--bin-sizes", default="1,3,10,30,100,1000")
    parser.add_argument("--positive-threshold", type=float, default=0.0)
    parser.add_argument(
        "--gtf",
        default=DEFAULT_GTF,
        help="Optional GTF for CDS/UTR boundary evaluation. Use 'none' to disable.",
    )
    parser.add_argument("--start-upstream", type=int, default=300)
    parser.add_argument("--start-downstream", type=int, default=600)
    parser.add_argument("--stop-upstream", type=int, default=300)
    parser.add_argument("--stop-downstream", type=int, default=600)
    parser.add_argument("--min-cds-bp", type=int, default=90)
    parser.add_argument("--min-utr5-bp", type=int, default=30)
    parser.add_argument("--min-utr3-bp", type=int, default=30)
    parser.add_argument(
        "--utr5-start-exclusion-bp",
        type=int,
        default=30,
        help=(
            "Exclude this many transcript-space 5'UTR bases immediately before "
            "the annotated start codon for the separate utr5_distal shape metrics."
        ),
    )
    parser.add_argument(
        "--filtered-transcript-min-obs-mean",
        type=float,
        default=0.0,
        help=(
            "Strict observed region mean cutoff for filtered per-transcript Pearson r; "
            "only transcript regions with observed mean greater than this value are included."
        ),
    )
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--max-rank-points",
        type=int,
        default=10000000,
        help="Maximum values to keep in memory for Spearman/AUROC/AUPRC/top-k metrics.",
    )
    return parser.parse_args()


def parse_ints(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_chroms(value):
    if value is None:
        return []

    value = value.strip()
    if not value or value.lower() in {"none", "null", "false", "0"}:
        return []

    chroms = [chrom.strip() for chrom in value.split(",") if chrom.strip()]
    if not chroms:
        raise ValueError(f"No chromosomes found in --chroms: {value}")
    return chroms


def load_regions(path):
    regions = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"Expected at least 3 BED columns, got: {line}")
            chrom, start, end = fields[:3]
            regions.append((chrom, int(start), int(end)))
    if not regions:
        raise ValueError(f"No regions found in {path}")
    return regions


def filter_regions_by_chroms(regions, chroms):
    allowed = set(chroms)
    filtered = [region for region in regions if region[0] in allowed]
    if not filtered:
        raise ValueError(f"No regions remain after filtering to chromosomes: {','.join(chroms)}")
    return filtered


def regions_by_chrom(regions):
    grouped = defaultdict(list)
    for chrom, start, end in regions:
        grouped[chrom].append((start, end))
    for chrom in grouped:
        grouped[chrom].sort()
    return dict(grouped)


def overlaps_any(chrom, start, end, grouped_regions):
    for region_start, region_end in grouped_regions.get(chrom, []):
        if start < region_end and end > region_start:
            return True
    return False


def point_in_regions(chrom, pos, grouped_regions):
    for region_start, region_end in grouped_regions.get(chrom, []):
        if region_start <= pos < region_end:
            return True
        if pos < region_start:
            return False
    return False


def parse_gtf_attributes(value):
    attributes = {}
    for item in value.rstrip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        if " " not in item:
            continue
        key, raw = item.split(" ", 1)
        attributes[key] = raw.strip().strip('"')
    return attributes


def open_text_maybe_gzip(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def intersect_intervals(intervals, mask_intervals):
    result = []
    mask_intervals = sorted(mask_intervals)
    for start, end in intervals:
        for mask_start, mask_end in mask_intervals:
            if mask_end <= start:
                continue
            if mask_start >= end:
                break
            clipped_start = max(start, mask_start)
            clipped_end = min(end, mask_end)
            if clipped_start < clipped_end:
                result.append((clipped_start, clipped_end))
    return merge_intervals(result)


def interval_length(intervals):
    return sum(end - start for start, end in intervals)


def interval_sum(bigwig, chrom, intervals):
    total = 0.0
    bases = 0
    for start, end in intervals:
        if end <= start:
            continue
        value = bigwig.stats(chrom, start, end, type="sum", exact=True)[0]
        total += 0.0 if value is None else float(value)
        bases += end - start
    return total, bases


def clip_to_regions(chrom, intervals, grouped_regions):
    return intersect_intervals(intervals, grouped_regions.get(chrom, []))


def load_gtf_transcripts(gtf_path, grouped_regions):
    transcripts = {}
    allowed_chroms = set(grouped_regions)

    with open_text_maybe_gzip(gtf_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            chrom, _source, feature, start, end, _score, strand, _frame, attrs = fields
            if chrom not in allowed_chroms:
                continue
            start = int(start) - 1
            end = int(end)
            parsed_attrs = parse_gtf_attributes(attrs)
            transcript_id = parsed_attrs.get("transcript_id")
            if transcript_id is None:
                continue

            transcript = transcripts.setdefault(
                transcript_id,
                {
                    "chrom": chrom,
                    "strand": strand,
                    "gene_id": parsed_attrs.get("gene_id", ""),
                    "gene_type": parsed_attrs.get("gene_type", ""),
                    "gene_name": parsed_attrs.get("gene_name", ""),
                    "transcript_id": transcript_id,
                    "transcript_type": parsed_attrs.get("transcript_type", ""),
                    "transcript_name": parsed_attrs.get("transcript_name", ""),
                    "transcript_start": start,
                    "transcript_end": end,
                    "exons": [],
                    "cds": [],
                    "start_codons": [],
                    "stop_codons": [],
                    "overlaps_regions": False,
                },
            )
            transcript["transcript_start"] = min(transcript["transcript_start"], start)
            transcript["transcript_end"] = max(transcript["transcript_end"], end)
            transcript["overlaps_regions"] = transcript["overlaps_regions"] or overlaps_any(
                chrom, start, end, grouped_regions
            )

            if feature == "exon":
                transcript["exons"].append((start, end))
            elif feature == "CDS":
                transcript["cds"].append((start, end))
            elif feature == "start_codon":
                transcript["start_codons"].append((start, end))
            elif feature == "stop_codon":
                transcript["stop_codons"].append((start, end))

    loaded = []
    for transcript in transcripts.values():
        transcript["exons"] = merge_intervals(transcript["exons"])
        transcript["cds"] = merge_intervals(transcript["cds"])
        transcript["start_codons"] = merge_intervals(transcript["start_codons"])
        transcript["stop_codons"] = merge_intervals(transcript["stop_codons"])
        if transcript["overlaps_regions"] and transcript["exons"]:
            loaded.append(transcript)
    return loaded


def utr5_intervals(transcript):
    exons = transcript["exons"]
    cds = transcript["cds"]
    if not cds:
        return []

    if transcript["strand"] == "+":
        boundary = min(start for start, _end in cds)
        utr_range = [(min(start for start, _end in exons), boundary)]
    else:
        boundary = max(end for _start, end in cds)
        utr_range = [(boundary, max(end for _start, end in exons))]

    return intersect_intervals(exons, utr_range)


def utr3_intervals(transcript):
    exons = transcript["exons"]
    cds = transcript["cds"]
    if not cds:
        return []

    if transcript["stop_codons"]:
        if transcript["strand"] == "+":
            boundary = max(end for _start, end in transcript["stop_codons"])
            utr_range = [(boundary, max(end for _start, end in exons))]
        else:
            boundary = min(start for start, _end in transcript["stop_codons"])
            utr_range = [(min(start for start, _end in exons), boundary)]
    elif transcript["strand"] == "+":
        boundary = max(end for _start, end in cds)
        utr_range = [(boundary, max(end for _start, end in exons))]
    else:
        boundary = min(start for start, _end in cds)
        utr_range = [(min(start for start, _end in exons), boundary)]

    return intersect_intervals(exons, utr_range)


def subtract_intervals(intervals, mask_intervals):
    remaining = []
    for start, end in sorted(intervals):
        cursor = start
        for mask_start, mask_end in sorted(mask_intervals):
            if mask_end <= cursor:
                continue
            if mask_start >= end:
                break
            if cursor < mask_start:
                remaining.append((cursor, min(mask_start, end)))
            cursor = max(cursor, mask_end)
            if cursor >= end:
                break
        if cursor < end:
            remaining.append((cursor, end))
    return merge_intervals(remaining)


def interval_mask(coords, intervals):
    mask = np.zeros(len(coords), dtype=bool)
    for start, end in intervals:
        mask |= (coords >= start) & (coords < end)
    return mask


def oriented_exon_arrays(transcript, bigwig, grouped_regions):
    chrom = transcript["chrom"]
    if transcript["strand"] == "+":
        exons = sorted(transcript["exons"])
        coord_chunks = [np.arange(start, end, dtype=np.int64) for start, end in exons]
        value_chunks = [bw_values(bigwig, chrom, start, end) for start, end in exons]
    else:
        exons = sorted(transcript["exons"], reverse=True)
        coord_chunks = [np.arange(end - 1, start - 1, -1, dtype=np.int64) for start, end in exons]
        value_chunks = [bw_values(bigwig, chrom, start, end)[::-1] for start, end in exons]

    if not coord_chunks:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    coords = np.concatenate(coord_chunks)
    values = np.concatenate(value_chunks).astype(np.float32)
    mask = np.fromiter(
        (point_in_regions(chrom, int(pos), grouped_regions) for pos in coords),
        dtype=bool,
        count=len(coords),
    )
    values = np.where(mask, values, np.nan)
    return coords, values


def stop_boundary_index(transcript, coords):
    if len(coords) == 0:
        return None

    stop_intervals = transcript["stop_codons"] or transcript["cds"]
    stop_mask = np.zeros(len(coords), dtype=bool)
    for start, end in stop_intervals:
        stop_mask |= (coords >= start) & (coords < end)
    indexes = np.flatnonzero(stop_mask)
    if len(indexes) == 0:
        return None
    return int(indexes.max() + 1)


def start_boundary_index(transcript, coords):
    if len(coords) == 0:
        return None

    cds_mask = interval_mask(coords, transcript["cds"])
    indexes = np.flatnonzero(cds_mask)
    if len(indexes) == 0:
        return None
    return int(indexes.min())


def profile_slice(values, boundary_index, upstream, downstream):
    width = upstream + downstream
    output = np.full(width, np.nan, dtype=np.float32)
    start = boundary_index - upstream
    end = boundary_index + downstream
    source_start = max(start, 0)
    source_end = min(end, len(values))
    if source_start >= source_end:
        return output
    dest_start = source_start - start
    output[dest_start:dest_start + (source_end - source_start)] = values[source_start:source_end]
    return output


def estimated_bins(regions, bin_size):
    return sum((end - start) // bin_size for _chrom, start, end in regions)


def bw_values(bigwig, chrom, start, end):
    values = np.asarray(bigwig.values(chrom, start, end), dtype=np.float32)
    if len(values) == 0 and end > start:
        values = np.zeros(end - start, dtype=np.float32)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def bin_values(values, bin_size):
    if bin_size == 1:
        return values
    trim_len = (len(values) // bin_size) * bin_size
    if trim_len == 0:
        return np.empty(0, dtype=np.float32)
    return values[:trim_len].reshape(-1, bin_size).sum(axis=1)


def safe_div(num, denom):
    return num / denom if denom else float("nan")


def pearson_from_sums(n, sum_x, sum_y, sum_x2, sum_y2, sum_xy):
    if n < 2:
        return float("nan")
    cov = sum_xy - (sum_x * sum_y / n)
    var_x = sum_x2 - (sum_x * sum_x / n)
    var_y = sum_y2 - (sum_y * sum_y / n)
    if var_x <= 0 or var_y <= 0:
        return float("nan")
    return cov / math.sqrt(var_x * var_y)


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman(x, y):
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(average_ranks(x), average_ranks(y))[0, 1])


def auroc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    pos = int(labels.sum())
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    rank_sum_pos = ranks[labels].sum()
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(len(sorted_labels)) + 1)
    return float(precision[sorted_labels].sum() / positives)


def safe_ratio(num, denom):
    if denom is None or denom == 0 or math.isnan(denom):
        return float("nan")
    return num / denom


def top_positive_metrics(labels, scores, fraction):
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    if len(labels) == 0 or positives == 0:
        return float("nan"), float("nan"), float("nan")

    top_n = max(1, int(math.ceil(len(labels) * fraction)))
    order = np.argsort(-scores, kind="mergesort")[:top_n]
    true_positives = int(labels[order].sum())
    precision = true_positives / top_n
    recall = true_positives / positives
    baseline = positives / len(labels)
    enrichment = safe_ratio(precision, baseline)
    return float(precision), float(recall), float(enrichment)


def max_f1(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        return float("nan"), float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    tp = np.cumsum(sorted_labels)
    rank = np.arange(len(sorted_labels)) + 1
    precision = tp / rank
    recall = tp / positives
    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(denom, dtype=np.float64),
        where=denom > 0,
    )
    best_idx = int(np.argmax(f1))
    return float(f1[best_idx]), float(sorted_scores[best_idx])


def top_signal_recovery(obs, pred, fraction):
    if len(obs) == 0:
        return float("nan")
    obs_sum = obs.sum()
    if obs_sum <= 0:
        return float("nan")
    top_n = max(1, int(math.ceil(len(obs) * fraction)))
    order = np.argsort(-pred, kind="mergesort")[:top_n]
    return float(obs[order].sum() / obs_sum)


class MetricAccumulator:
    def __init__(self, positive_threshold, keep_values):
        self.positive_threshold = positive_threshold
        self.keep_values = keep_values
        self.n = 0
        self.sum_obs = 0.0
        self.sum_pred = 0.0
        self.sum_obs2 = 0.0
        self.sum_pred2 = 0.0
        self.sum_obs_pred = 0.0
        self.sum_log_obs = 0.0
        self.sum_log_pred = 0.0
        self.sum_log_obs2 = 0.0
        self.sum_log_pred2 = 0.0
        self.sum_log_obs_log_pred = 0.0
        self.sum_abs_error = 0.0
        self.sum_sq_error = 0.0
        self.sum_poisson_nll = 0.0
        self.positive_count = 0
        self.obs_chunks = []
        self.pred_chunks = []

    def add(self, obs, pred):
        obs = np.asarray(obs, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        if len(obs) != len(pred):
            raise ValueError(f"obs/pred length mismatch: {len(obs)} != {len(pred)}")
        if len(obs) == 0:
            return

        pred = np.clip(pred, 0.0, None)
        log_obs = np.log1p(obs)
        log_pred = np.log1p(pred)
        error = pred - obs
        positive = obs > self.positive_threshold
        pred_for_poisson = np.clip(pred, 1e-8, None)

        self.n += len(obs)
        self.sum_obs += obs.sum()
        self.sum_pred += pred.sum()
        self.sum_obs2 += np.square(obs).sum()
        self.sum_pred2 += np.square(pred).sum()
        self.sum_obs_pred += (obs * pred).sum()
        self.sum_log_obs += log_obs.sum()
        self.sum_log_pred += log_pred.sum()
        self.sum_log_obs2 += np.square(log_obs).sum()
        self.sum_log_pred2 += np.square(log_pred).sum()
        self.sum_log_obs_log_pred += (log_obs * log_pred).sum()
        self.sum_abs_error += np.abs(error).sum()
        self.sum_sq_error += np.square(error).sum()
        self.sum_poisson_nll += (pred_for_poisson - obs * np.log(pred_for_poisson)).sum()
        self.positive_count += int(positive.sum())

        if self.keep_values:
            self.obs_chunks.append(obs.astype(np.float32))
            self.pred_chunks.append(pred.astype(np.float32))

    def summarize(self):
        pearson = pearson_from_sums(
            self.n,
            self.sum_obs,
            self.sum_pred,
            self.sum_obs2,
            self.sum_pred2,
            self.sum_obs_pred,
        )
        log1p_pearson = pearson_from_sums(
            self.n,
            self.sum_log_obs,
            self.sum_log_pred,
            self.sum_log_obs2,
            self.sum_log_pred2,
            self.sum_log_obs_log_pred,
        )

        summary = {
            "n": self.n,
            "obs_mean": safe_div(self.sum_obs, self.n),
            "pred_mean": safe_div(self.sum_pred, self.n),
            "obs_sum": self.sum_obs,
            "pred_sum": self.sum_pred,
            "pearson": pearson,
            "log1p_pearson": log1p_pearson,
            "mae": safe_div(self.sum_abs_error, self.n),
            "rmse": math.sqrt(safe_div(self.sum_sq_error, self.n)),
            "poisson_nll_no_constant": safe_div(self.sum_poisson_nll, self.n),
            "positive_fraction": safe_div(self.positive_count, self.n),
        }

        if self.keep_values and self.obs_chunks:
            obs = np.concatenate(self.obs_chunks)
            pred = np.concatenate(self.pred_chunks)
            labels = obs > self.positive_threshold
            auprc_value = average_precision(labels, pred)
            positive_fraction = safe_div(int(labels.sum()), len(labels))
            precision_1, recall_1, enrichment_1 = top_positive_metrics(labels, pred, 0.01)
            precision_5, recall_5, enrichment_5 = top_positive_metrics(labels, pred, 0.05)
            precision_10, recall_10, enrichment_10 = top_positive_metrics(labels, pred, 0.10)
            best_f1, best_f1_threshold = max_f1(labels, pred)
            summary.update({
                "spearman": spearman(obs, pred),
                "auroc_nonzero": auroc(labels, pred),
                "auprc_nonzero": auprc_value,
                "auprc_baseline": positive_fraction,
                "auprc_lift": safe_ratio(auprc_value, positive_fraction),
                "max_f1_nonzero": best_f1,
                "max_f1_threshold": best_f1_threshold,
                "top_1pct_precision_nonzero": precision_1,
                "top_1pct_recall_nonzero": recall_1,
                "top_1pct_enrichment_nonzero": enrichment_1,
                "top_5pct_precision_nonzero": precision_5,
                "top_5pct_recall_nonzero": recall_5,
                "top_5pct_enrichment_nonzero": enrichment_5,
                "top_10pct_precision_nonzero": precision_10,
                "top_10pct_recall_nonzero": recall_10,
                "top_10pct_enrichment_nonzero": enrichment_10,
                "top_1pct_signal_recovery": top_signal_recovery(obs, pred, 0.01),
                "top_5pct_signal_recovery": top_signal_recovery(obs, pred, 0.05),
                "top_10pct_signal_recovery": top_signal_recovery(obs, pred, 0.10),
            })
        else:
            summary.update({
                "spearman": float("nan"),
                "auroc_nonzero": float("nan"),
                "auprc_nonzero": float("nan"),
                "auprc_baseline": float("nan"),
                "auprc_lift": float("nan"),
                "max_f1_nonzero": float("nan"),
                "max_f1_threshold": float("nan"),
                "top_1pct_precision_nonzero": float("nan"),
                "top_1pct_recall_nonzero": float("nan"),
                "top_1pct_enrichment_nonzero": float("nan"),
                "top_5pct_precision_nonzero": float("nan"),
                "top_5pct_recall_nonzero": float("nan"),
                "top_5pct_enrichment_nonzero": float("nan"),
                "top_10pct_precision_nonzero": float("nan"),
                "top_10pct_recall_nonzero": float("nan"),
                "top_10pct_enrichment_nonzero": float("nan"),
                "top_1pct_signal_recovery": float("nan"),
                "top_5pct_signal_recovery": float("nan"),
                "top_10pct_signal_recovery": float("nan"),
            })

        return summary


class PositionCorrelationAccumulator:
    def __init__(self):
        self.n = 0
        self.sum_obs = 0.0
        self.sum_pred = 0.0
        self.sum_obs2 = 0.0
        self.sum_pred2 = 0.0
        self.sum_obs_pred = 0.0
        self.sum_log_obs = 0.0
        self.sum_log_pred = 0.0
        self.sum_log_obs2 = 0.0
        self.sum_log_pred2 = 0.0
        self.sum_log_obs_log_pred = 0.0

    def add(self, obs, pred):
        obs = np.asarray(obs, dtype=np.float64)
        pred = np.clip(np.asarray(pred, dtype=np.float64), 0.0, None)
        valid = np.isfinite(obs) & np.isfinite(pred)
        if not valid.any():
            return

        obs = obs[valid]
        pred = pred[valid]
        log_obs = np.log1p(obs)
        log_pred = np.log1p(pred)

        self.n += len(obs)
        self.sum_obs += obs.sum()
        self.sum_pred += pred.sum()
        self.sum_obs2 += np.square(obs).sum()
        self.sum_pred2 += np.square(pred).sum()
        self.sum_obs_pred += (obs * pred).sum()
        self.sum_log_obs += log_obs.sum()
        self.sum_log_pred += log_pred.sum()
        self.sum_log_obs2 += np.square(log_obs).sum()
        self.sum_log_pred2 += np.square(log_pred).sum()
        self.sum_log_obs_log_pred += (log_obs * log_pred).sum()

    def summarize(self):
        return {
            "n_bases": self.n,
            "obs_mean": safe_div(self.sum_obs, self.n),
            "pred_mean": safe_div(self.sum_pred, self.n),
            "pearson": pearson_from_sums(
                self.n,
                self.sum_obs,
                self.sum_pred,
                self.sum_obs2,
                self.sum_pred2,
                self.sum_obs_pred,
            ),
            "log1p_pearson": pearson_from_sums(
                self.n,
                self.sum_log_obs,
                self.sum_log_pred,
                self.sum_log_obs2,
                self.sum_log_pred2,
                self.sum_log_obs_log_pred,
            ),
        }


def position_correlations(obs, pred):
    acc = PositionCorrelationAccumulator()
    acc.add(obs, pred)
    summary = acc.summarize()
    return summary["n_bases"], summary["pearson"], summary["log1p_pearson"]


def make_accumulators(bin_size, regions, max_rank_points, positive_threshold):
    keep_values = estimated_bins(regions, bin_size) <= max_rank_points
    return {
        "plus": MetricAccumulator(positive_threshold, keep_values),
        "minus": MetricAccumulator(positive_threshold, keep_values),
        "combined": MetricAccumulator(positive_threshold, keep_values),
        "pred_plus_vs_obs_minus": MetricAccumulator(positive_threshold, keep_values),
        "pred_minus_vs_obs_plus": MetricAccumulator(positive_threshold, keep_values),
        "pred_plus_vs_pred_minus": MetricAccumulator(positive_threshold, keep_values),
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def frame_sums(values, mask, reference_index):
    valid = np.isfinite(values) & mask
    sums = np.zeros(3, dtype=np.float64)
    if not valid.any():
        return sums
    indexes = np.flatnonzero(valid)
    frames = (indexes - reference_index) % 3
    for frame in range(3):
        sums[frame] = float(np.nansum(values[indexes[frames == frame]]))
    return sums


def dominant_frame(sums):
    total = float(np.sum(sums))
    if total <= 0:
        return None
    return int(np.argmax(sums))


def frame_fraction(sums, frame):
    total = float(np.sum(sums))
    return safe_div(float(sums[frame]), total)


def transcript_start_codon_reference(transcript, coords, cds_coord_mask):
    if transcript.get("start_codons"):
        start_mask = interval_mask(coords, transcript["start_codons"])
        start_indexes = np.flatnonzero(start_mask)
        if len(start_indexes):
            return int(start_indexes.min()), "start_codon"

    cds_indexes = np.flatnonzero(cds_coord_mask)
    if len(cds_indexes):
        return int(cds_indexes.min()), "cds_start"
    return None, ""


def distal_utr5_mask(utr5_mask, start_codon_index, exclusion_bp):
    transcript_indexes = np.arange(len(utr5_mask), dtype=np.int64)
    return utr5_mask & (transcript_indexes < start_codon_index - exclusion_bp)


def add_start_codon_frame_fields(row, prefix, sums):
    total = float(np.sum(sums))
    dominant = dominant_frame(sums)
    row[f"{prefix}_sum"] = total
    row[f"{prefix}_dominant_frame"] = "" if dominant is None else dominant
    for frame in range(3):
        row[f"{prefix}_frame{frame}_sum"] = float(sums[frame])
        row[f"{prefix}_frame{frame}_fraction"] = frame_fraction(sums, frame)


def start_codon_frame_fields():
    fields = [
        "chrom",
        "gene_id",
        "gene_type",
        "gene_name",
        "transcript_id",
        "transcript_type",
        "transcript_name",
        "strand",
        "transcript_start",
        "transcript_end",
        "start_codon_tx_index",
        "start_codon_genomic_1based",
        "start_codon_reference",
        "utr5_start_exclusion_bp",
        "expression_bin",
    ]
    fields.extend(f"{region}_bp" for region in START_CODON_FRAME_REGIONS)

    for prefix, _source in START_CODON_FRAME_SOURCES:
        for region in START_CODON_FRAME_REGIONS:
            base = f"{prefix}_{region}"
            fields.extend([
                f"{base}_sum",
                f"{base}_dominant_frame",
            ])
            for frame in range(3):
                fields.extend([
                    f"{base}_frame{frame}_sum",
                    f"{base}_frame{frame}_fraction",
                ])
    return fields


def start_codon_frame_summary_fields():
    fields = [
        "category",
        "expression_bin",
        "strand",
        "source",
        "transcripts",
        "bases",
        "total_signal",
        "dominant_frame",
    ]
    for frame in range(3):
        fields.extend([
            f"frame{frame}_sum",
            f"pooled_frame{frame}_fraction",
            f"mean_frame{frame}_fraction",
        ])
    fields.append("frame0_vs_other_fraction_delta")
    return fields


def summarize_start_codon_frame_rows(rows):
    output = []
    expression_bins = ("all", "zero_observed", "low", "medium", "high")
    strands = ("all", "+", "-")

    for category in START_CODON_FRAME_REGIONS:
        for expression_bin in expression_bins:
            for strand in strands:
                group = [
                    row for row in rows
                    if metric_value(row, f"{category}_bp") > 0
                    and (expression_bin == "all" or row["expression_bin"] == expression_bin)
                    and (strand == "all" or row["strand"] == strand)
                ]
                if not group:
                    continue

                bases = int(sum(metric_value(row, f"{category}_bp") for row in group))
                for prefix, source in START_CODON_FRAME_SOURCES:
                    frame_totals = [
                        sum(
                            metric_value(row, f"{prefix}_{category}_frame{frame}_sum")
                            for row in group
                        )
                        for frame in range(3)
                    ]
                    total_signal = float(sum(frame_totals))
                    pooled_fractions = [
                        safe_div(frame_total, total_signal)
                        for frame_total in frame_totals
                    ]
                    row = {
                        "category": category,
                        "expression_bin": expression_bin,
                        "strand": strand,
                        "source": source,
                        "transcripts": len(group),
                        "bases": bases,
                        "total_signal": total_signal,
                        "dominant_frame": (
                            "" if total_signal <= 0 else int(np.argmax(frame_totals))
                        ),
                    }
                    for frame in range(3):
                        row[f"frame{frame}_sum"] = float(frame_totals[frame])
                        row[f"pooled_frame{frame}_fraction"] = pooled_fractions[frame]
                        row[f"mean_frame{frame}_fraction"] = finite_mean(
                            metric_value(
                                transcript_row,
                                f"{prefix}_{category}_frame{frame}_fraction",
                            )
                            for transcript_row in group
                        )
                    row["frame0_vs_other_fraction_delta"] = (
                        pooled_fractions[0]
                        - ((pooled_fractions[1] + pooled_fractions[2]) / 2.0)
                        if all(math.isfinite(value) for value in pooled_fractions)
                        else float("nan")
                    )
                    output.append(row)

    return output


def quantile_expression_bin(value, lower, upper):
    if math.isnan(value):
        return "unknown"
    if value <= lower:
        return "low"
    if value <= upper:
        return "medium"
    return "high"


def finite_mean(values):
    values = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    return safe_div(sum(values), len(values))


def finite_array(values):
    output = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            output.append(value)
    return np.asarray(output, dtype=np.float64)


def finite_median(values):
    values = finite_array(values)
    if len(values) == 0:
        return float("nan")
    return float(np.median(values))


def finite_weighted_mean(values, weights):
    pairs = []
    for value, weight in zip(values, weights):
        try:
            value = float(value)
            weight = float(weight)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and math.isfinite(weight) and weight > 0:
            pairs.append((value, weight))
    if not pairs:
        return float("nan")

    values = np.asarray([value for value, _weight in pairs], dtype=np.float64)
    weights = np.asarray([weight for _value, weight in pairs], dtype=np.float64)
    return float(np.average(values, weights=weights))


def fisher_mean_r(values):
    values = finite_array(values)
    if len(values) == 0:
        return float("nan")
    values = np.clip(values, -0.999999, 0.999999)
    return float(np.tanh(np.arctanh(values).mean()))


def transcript_region_metrics(obs_values, pred_values, mask):
    valid = np.isfinite(obs_values) & np.isfinite(pred_values) & mask
    obs = obs_values[valid]
    pred = pred_values[valid]
    n, pearson, log1p_pearson = position_correlations(obs, pred)
    return {
        "bp": int(len(obs)),
        "obs_sum": float(obs.sum()) if len(obs) else 0.0,
        "pred_sum": float(pred.sum()) if len(pred) else 0.0,
        "obs_mean": safe_div(float(obs.sum()), len(obs)),
        "pred_mean": safe_div(float(pred.sum()), len(pred)),
        "position_n": n,
        "position_pearson": pearson,
        "position_log1p_pearson": log1p_pearson,
    }


def normalized_profile_correlation(obs_values, pred_values, mask):
    valid = np.isfinite(obs_values) & np.isfinite(pred_values) & mask
    obs = obs_values[valid].astype(np.float64)
    pred = pred_values[valid].astype(np.float64)
    if len(obs) < 2:
        return 0, float("nan")

    obs_total = float(obs.sum())
    pred_total = float(pred.sum())
    if obs_total <= 0 or pred_total <= 0:
        return len(obs), float("nan")

    obs_profile = obs / obs_total
    pred_profile = pred / pred_total
    n, pearson, _log1p_pearson = position_correlations(obs_profile, pred_profile)
    return n, pearson


def transcript_codon_metrics(obs_values, pred_values, mask, start_codon_index):
    valid = np.isfinite(obs_values) & np.isfinite(pred_values) & mask
    indexes = np.flatnonzero(valid)
    if len(indexes) < 3:
        return {
            "n": 0,
            "pearson": float("nan"),
            "log1p_pearson": float("nan"),
            "obs": np.empty(0, dtype=np.float64),
            "pred": np.empty(0, dtype=np.float64),
        }

    relative_positions = indexes - start_codon_index
    codon_indexes = np.floor_divide(relative_positions, 3)
    obs_codons = []
    pred_codons = []
    for codon_index in np.unique(codon_indexes):
        members = indexes[codon_indexes == codon_index]
        expected = start_codon_index + codon_index * 3 + np.arange(3)
        if len(members) != 3 or not np.array_equal(members, expected):
            continue
        obs_codons.append(float(np.sum(obs_values[members])))
        pred_codons.append(float(np.sum(pred_values[members])))

    obs = np.asarray(obs_codons, dtype=np.float64)
    pred = np.asarray(pred_codons, dtype=np.float64)
    n, pearson, log1p_pearson = position_correlations(obs, pred)
    return {
        "n": n,
        "pearson": pearson,
        "log1p_pearson": log1p_pearson,
        "obs": obs,
        "pred": pred,
    }


def add_region_metrics_to_row(row, region, metrics):
    row[f"{region}_bp"] = metrics["bp"]
    row[f"obs_{region}_sum"] = metrics["obs_sum"]
    row[f"pred_{region}_sum"] = metrics["pred_sum"]
    row[f"obs_{region}_mean"] = metrics["obs_mean"]
    row[f"pred_{region}_mean"] = metrics["pred_mean"]
    row[f"{region}_position_n"] = metrics["position_n"]
    row[f"{region}_position_pearson"] = metrics["position_pearson"]
    row[f"{region}_position_log1p_pearson"] = metrics["position_log1p_pearson"]


def add_codon_metrics_to_row(row, region, metrics):
    row[f"{region}_codon_n"] = metrics["n"]
    row[f"{region}_codon_pearson"] = metrics["pearson"]
    row[f"{region}_codon_log1p_pearson"] = metrics["log1p_pearson"]


def rank_descending(values):
    order = np.argsort(-np.asarray(values), kind="mergesort")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def add_expression_units(rows):
    obs_total = sum(metric_value(row, "obs_cds_sum") for row in rows)
    pred_total = sum(metric_value(row, "pred_cds_sum") for row in rows)
    obs_signal_per_kb = []
    pred_signal_per_kb = []
    for row in rows:
        kb = safe_div(metric_value(row, "cds_bp"), 1000.0)
        obs_signal_per_kb.append(safe_div(metric_value(row, "obs_cds_sum"), kb))
        pred_signal_per_kb.append(safe_div(metric_value(row, "pred_cds_sum"), kb))

    obs_signal_per_kb_total = sum(
        value for value in obs_signal_per_kb if not math.isnan(value)
    )
    pred_signal_per_kb_total = sum(
        value for value in pred_signal_per_kb if not math.isnan(value)
    )
    for row, obs_per_kb_value, pred_per_kb_value in zip(
        rows,
        obs_signal_per_kb,
        pred_signal_per_kb,
    ):
        row["obs_cds_signal_ppm"] = safe_div(
            metric_value(row, "obs_cds_sum") * 1e6,
            obs_total,
        )
        row["pred_cds_signal_ppm"] = safe_div(
            metric_value(row, "pred_cds_sum") * 1e6,
            pred_total,
        )
        row["obs_cds_signal_per_kb_ppm"] = safe_div(
            obs_per_kb_value * 1e6,
            obs_signal_per_kb_total,
        )
        row["pred_cds_signal_per_kb_ppm"] = safe_div(
            pred_per_kb_value * 1e6,
            pred_signal_per_kb_total,
        )

    for metric in ("cds_signal_ppm", "cds_signal_per_kb_ppm"):
        obs_values = [metric_value(row, f"obs_{metric}") for row in rows]
        pred_values = [metric_value(row, f"pred_{metric}") for row in rows]
        obs_ranks = rank_descending(obs_values)
        pred_ranks = rank_descending(pred_values)
        for row, obs_rank, pred_rank in zip(rows, obs_ranks, pred_ranks):
            row[f"obs_{metric}_rank"] = int(obs_rank)
            row[f"pred_{metric}_rank"] = int(pred_rank)


def topk_overlap_rows(rows, top_sizes=(50, 100, 500)):
    output = []
    for metric in ("cds_signal_ppm", "cds_signal_per_kb_ppm"):
        for top_n in top_sizes:
            available = len(rows)
            used_n = min(top_n, available)
            if used_n == 0:
                continue
            obs_top = {
                row["transcript_id"]
                for row in rows
                if int(row[f"obs_{metric}_rank"]) <= used_n
            }
            pred_top = {
                row["transcript_id"]
                for row in rows
                if int(row[f"pred_{metric}_rank"]) <= used_n
            }
            overlap = obs_top & pred_top
            obs_top_rows = sorted(
                (row for row in rows if row["transcript_id"] in obs_top),
                key=lambda row: int(row[f"obs_{metric}_rank"]),
            )
            pred_top_rows = sorted(
                (row for row in rows if row["transcript_id"] in pred_top),
                key=lambda row: int(row[f"pred_{metric}_rank"]),
            )
            output.append({
                "metric": metric,
                "requested_top_n": top_n,
                "used_top_n": used_n,
                "available_transcripts": available,
                "overlap_count": len(overlap),
                "overlap_fraction": safe_div(len(overlap), used_n),
                "obs_top_genes": ";".join(row["gene_name"] or row["transcript_id"] for row in obs_top_rows[:20]),
                "pred_top_genes": ";".join(row["gene_name"] or row["transcript_id"] for row in pred_top_rows[:20]),
                "overlap_genes": ";".join(
                    sorted(
                        row["gene_name"] or row["transcript_id"]
                        for row in rows
                        if row["transcript_id"] in overlap
                    )[:50]
                ),
            })
    return output


def filtered_per_transcript_r_rows(phase_rows, min_obs_cds_mean):
    output = []
    expression_bins = ("all", "zero_observed", "low", "medium", "high")
    strands = ("all", "+", "-")

    for expression_bin in expression_bins:
        for strand in strands:
            group = [
                row for row in phase_rows
                if (expression_bin == "all" or row["expression_bin"] == expression_bin)
                and (strand == "all" or row["strand"] == strand)
            ]
            if not group:
                continue

            coverage_passing = [
                row for row in group
                if metric_value(row, "obs_cds_mean") > min_obs_cds_mean
            ]
            pearson_rows = [
                row for row in coverage_passing
                if math.isfinite(metric_value(row, "cds_position_pearson"))
            ]
            log1p_pearson_rows = [
                row for row in coverage_passing
                if math.isfinite(metric_value(row, "cds_position_log1p_pearson"))
            ]
            pearsons = [metric_value(row, "cds_position_pearson") for row in pearson_rows]
            log1p_pearsons = [
                metric_value(row, "cds_position_log1p_pearson")
                for row in log1p_pearson_rows
            ]
            profile_rows = [
                row for row in coverage_passing
                if math.isfinite(metric_value(row, "cds_profile_pearson"))
            ]
            profile_pearsons = [
                metric_value(row, "cds_profile_pearson")
                for row in profile_rows
            ]

            output.append({
                "expression_bin": expression_bin,
                "strand": strand,
                "min_obs_cds_mean_exclusive": min_obs_cds_mean,
                "transcripts": len(group),
                "coverage_passing_transcripts": len(coverage_passing),
                "pearson_callable_transcripts": len(pearson_rows),
                "log1p_pearson_callable_transcripts": len(log1p_pearson_rows),
                "profile_pearson_callable_transcripts": len(profile_rows),
                "mean_obs_cds_mean": finite_mean(
                    metric_value(row, "obs_cds_mean") for row in coverage_passing
                ),
                "median_obs_cds_mean": finite_median(
                    metric_value(row, "obs_cds_mean") for row in coverage_passing
                ),
                "mean_pred_cds_mean": finite_mean(
                    metric_value(row, "pred_cds_mean") for row in coverage_passing
                ),
                "median_pred_cds_mean": finite_median(
                    metric_value(row, "pred_cds_mean") for row in coverage_passing
                ),
                "mean_r": finite_mean(pearsons),
                "median_r": finite_median(pearsons),
                "fisher_mean_r": fisher_mean_r(pearsons),
                "cds_bp_weighted_mean_r": finite_weighted_mean(
                    pearsons,
                    (metric_value(row, "cds_bp") for row in pearson_rows),
                ),
                "mean_log1p_r": finite_mean(log1p_pearsons),
                "median_log1p_r": finite_median(log1p_pearsons),
                "fisher_mean_log1p_r": fisher_mean_r(log1p_pearsons),
                "cds_bp_weighted_mean_log1p_r": finite_weighted_mean(
                    log1p_pearsons,
                    (metric_value(row, "cds_bp") for row in log1p_pearson_rows),
                ),
                "mean_profile_r": finite_mean(profile_pearsons),
                "median_profile_r": finite_median(profile_pearsons),
                "fisher_mean_profile_r": fisher_mean_r(profile_pearsons),
                "cds_bp_weighted_mean_profile_r": finite_weighted_mean(
                    profile_pearsons,
                    (metric_value(row, "cds_bp") for row in profile_rows),
                ),
            })

    return output


def filtered_per_transcript_region_r_rows(phase_rows, min_obs_mean):
    output = []
    expression_bins = ("all", "zero_observed", "low", "medium", "high")
    strands = ("all", "+", "-")

    for region in CORRELATION_REGIONS:
        bp_field = f"{region}_bp"
        obs_mean_field = f"obs_{region}_mean"
        pred_mean_field = f"pred_{region}_mean"
        pearson_field = f"{region}_position_pearson"
        log1p_pearson_field = f"{region}_position_log1p_pearson"

        for expression_bin in expression_bins:
            for strand in strands:
                group = [
                    row for row in phase_rows
                    if metric_value(row, bp_field) > 0
                    and (expression_bin == "all" or row["expression_bin"] == expression_bin)
                    and (strand == "all" or row["strand"] == strand)
                ]
                if not group:
                    continue

                coverage_passing = [
                    row for row in group
                    if metric_value(row, obs_mean_field) > min_obs_mean
                ]
                pearson_rows = [
                    row for row in coverage_passing
                    if math.isfinite(metric_value(row, pearson_field))
                ]
                log1p_pearson_rows = [
                    row for row in coverage_passing
                    if math.isfinite(metric_value(row, log1p_pearson_field))
                ]
                pearsons = [metric_value(row, pearson_field) for row in pearson_rows]
                log1p_pearsons = [
                    metric_value(row, log1p_pearson_field)
                    for row in log1p_pearson_rows
                ]

                output.append({
                    "category": region,
                    "expression_bin": expression_bin,
                    "strand": strand,
                    "min_obs_mean_exclusive": min_obs_mean,
                    "transcripts": len(group),
                    "coverage_passing_transcripts": len(coverage_passing),
                    "pearson_callable_transcripts": len(pearson_rows),
                    "log1p_pearson_callable_transcripts": len(log1p_pearson_rows),
                    "mean_obs_mean": finite_mean(
                        metric_value(row, obs_mean_field) for row in coverage_passing
                    ),
                    "median_obs_mean": finite_median(
                        metric_value(row, obs_mean_field) for row in coverage_passing
                    ),
                    "mean_pred_mean": finite_mean(
                        metric_value(row, pred_mean_field) for row in coverage_passing
                    ),
                    "median_pred_mean": finite_median(
                        metric_value(row, pred_mean_field) for row in coverage_passing
                    ),
                    "mean_r": finite_mean(pearsons),
                    "median_r": finite_median(pearsons),
                    "fisher_mean_r": fisher_mean_r(pearsons),
                    "bp_weighted_mean_r": finite_weighted_mean(
                        pearsons,
                        (metric_value(row, bp_field) for row in pearson_rows),
                    ),
                    "mean_log1p_r": finite_mean(log1p_pearsons),
                    "median_log1p_r": finite_median(log1p_pearsons),
                    "fisher_mean_log1p_r": fisher_mean_r(log1p_pearsons),
                    "bp_weighted_mean_log1p_r": finite_weighted_mean(
                        log1p_pearsons,
                        (metric_value(row, bp_field) for row in log1p_pearson_rows),
                    ),
                })

    return output


def filtered_per_transcript_codon_r_rows(phase_rows, min_obs_mean):
    output = []
    expression_bins = ("all", "zero_observed", "low", "medium", "high")
    strands = ("all", "+", "-")

    for region in CORRELATION_REGIONS:
        bp_field = f"{region}_bp"
        obs_mean_field = f"obs_{region}_mean"
        pred_mean_field = f"pred_{region}_mean"
        codon_n_field = f"{region}_codon_n"
        pearson_field = f"{region}_codon_pearson"
        log1p_pearson_field = f"{region}_codon_log1p_pearson"

        for expression_bin in expression_bins:
            for strand in strands:
                group = [
                    row for row in phase_rows
                    if metric_value(row, bp_field) > 0
                    and (expression_bin == "all" or row["expression_bin"] == expression_bin)
                    and (strand == "all" or row["strand"] == strand)
                ]
                if not group:
                    continue

                coverage_passing = [
                    row for row in group
                    if metric_value(row, obs_mean_field) > min_obs_mean
                ]
                pearson_rows = [
                    row for row in coverage_passing
                    if math.isfinite(metric_value(row, pearson_field))
                ]
                log1p_pearson_rows = [
                    row for row in coverage_passing
                    if math.isfinite(metric_value(row, log1p_pearson_field))
                ]
                pearsons = [metric_value(row, pearson_field) for row in pearson_rows]
                log1p_pearsons = [
                    metric_value(row, log1p_pearson_field)
                    for row in log1p_pearson_rows
                ]

                output.append({
                    "category": region,
                    "expression_bin": expression_bin,
                    "strand": strand,
                    "min_obs_mean_exclusive": min_obs_mean,
                    "transcripts": len(group),
                    "coverage_passing_transcripts": len(coverage_passing),
                    "pearson_callable_transcripts": len(pearson_rows),
                    "log1p_pearson_callable_transcripts": len(log1p_pearson_rows),
                    "mean_obs_mean": finite_mean(
                        metric_value(row, obs_mean_field) for row in coverage_passing
                    ),
                    "median_obs_mean": finite_median(
                        metric_value(row, obs_mean_field) for row in coverage_passing
                    ),
                    "mean_pred_mean": finite_mean(
                        metric_value(row, pred_mean_field) for row in coverage_passing
                    ),
                    "median_pred_mean": finite_median(
                        metric_value(row, pred_mean_field) for row in coverage_passing
                    ),
                    "mean_r": finite_mean(pearsons),
                    "median_r": finite_median(pearsons),
                    "fisher_mean_r": fisher_mean_r(pearsons),
                    "codon_weighted_mean_r": finite_weighted_mean(
                        pearsons,
                        (metric_value(row, codon_n_field) for row in pearson_rows),
                    ),
                    "mean_log1p_r": finite_mean(log1p_pearsons),
                    "median_log1p_r": finite_median(log1p_pearsons),
                    "fisher_mean_log1p_r": fisher_mean_r(log1p_pearsons),
                    "codon_weighted_mean_log1p_r": finite_weighted_mean(
                        log1p_pearsons,
                        (metric_value(row, codon_n_field) for row in log1p_pearson_rows),
                    ),
                })

    return output


def evaluate_phase_controls(
    transcripts,
    grouped_regions,
    observed_plus,
    observed_minus,
    predicted_plus,
    predicted_minus,
    min_cds_bp,
    utr5_start_exclusion_bp,
):
    if utr5_start_exclusion_bp < 0:
        raise ValueError("--utr5-start-exclusion-bp must be non-negative")

    phase_rows = []
    start_codon_frame_rows = []

    transcript_payloads = []
    noncoding_count = 0
    region_payloads = []
    region_codon_payloads = []

    for transcript in tqdm(transcripts, desc="Evaluating phase controls"):
        chrom = transcript["chrom"]
        strand = transcript["strand"]
        obs_bw = observed_plus if strand == "+" else observed_minus
        pred_bw = predicted_plus if strand == "+" else predicted_minus

        obs_coords, obs_values = oriented_exon_arrays(transcript, obs_bw, grouped_regions)
        pred_coords, pred_values = oriented_exon_arrays(transcript, pred_bw, grouped_regions)
        if len(obs_coords) == 0 or len(obs_coords) != len(pred_coords):
            continue
        finite = np.isfinite(obs_values) & np.isfinite(pred_values)

        if not transcript["cds"]:
            noncoding_count += 1
            continue

        cds_coord_mask = interval_mask(obs_coords, transcript["cds"])
        utr5_coord_mask = interval_mask(obs_coords, utr5_intervals(transcript))
        utr3_coord_mask = interval_mask(obs_coords, utr3_intervals(transcript))
        cds_mask = cds_coord_mask & finite
        utr5_mask = utr5_coord_mask & finite
        utr3_mask = utr3_coord_mask & finite
        cds_metrics = transcript_region_metrics(obs_values, pred_values, cds_mask)
        cds_bp = cds_metrics["bp"]
        if cds_bp < min_cds_bp:
            continue

        start_codon_index, start_codon_source = transcript_start_codon_reference(
            transcript,
            obs_coords,
            cds_coord_mask,
        )
        if start_codon_index is None:
            continue

        utr5_distal_mask = distal_utr5_mask(
            utr5_mask,
            start_codon_index,
            utr5_start_exclusion_bp,
        )
        region_masks = {
            "cds": cds_mask,
            "utr5": utr5_mask,
            "utr5_distal": utr5_distal_mask,
            "utr3": utr3_mask,
        }
        region_metrics = {
            region: transcript_region_metrics(obs_values, pred_values, mask)
            for region, mask in region_masks.items()
        }

        obs_frame_sums = frame_sums(obs_values, cds_mask, start_codon_index)
        pred_frame_sums = frame_sums(pred_values, cds_mask, start_codon_index)
        obs_total = region_metrics["cds"]["obs_sum"]
        pred_total = region_metrics["cds"]["pred_sum"]
        cds_profile_n, cds_profile_pearson = normalized_profile_correlation(
            obs_values,
            pred_values,
            cds_mask,
        )
        obs_dom = dominant_frame(obs_frame_sums)
        pred_dom = dominant_frame(pred_frame_sums)
        obs_cds_mean = region_metrics["cds"]["obs_mean"]
        pred_cds_mean = region_metrics["cds"]["pred_mean"]

        row = {
            "chrom": chrom,
            "gene_id": transcript["gene_id"],
            "gene_type": transcript["gene_type"],
            "gene_name": transcript["gene_name"],
            "transcript_id": transcript["transcript_id"],
            "transcript_type": transcript["transcript_type"],
            "transcript_name": transcript["transcript_name"],
            "strand": strand,
            "transcript_start": transcript["transcript_start"],
            "transcript_end": transcript["transcript_end"],
            "cds_profile_n": cds_profile_n,
            "cds_profile_pearson": cds_profile_pearson,
            "obs_dominant_frame": "" if obs_dom is None else obs_dom,
            "pred_dominant_frame": "" if pred_dom is None else pred_dom,
            "phase_match": int(obs_dom == pred_dom) if obs_dom is not None and pred_dom is not None else "",
        }
        start_codon_frame_row = {
            "chrom": chrom,
            "gene_id": transcript["gene_id"],
            "gene_type": transcript["gene_type"],
            "gene_name": transcript["gene_name"],
            "transcript_id": transcript["transcript_id"],
            "transcript_type": transcript["transcript_type"],
            "transcript_name": transcript["transcript_name"],
            "strand": strand,
            "transcript_start": transcript["transcript_start"],
            "transcript_end": transcript["transcript_end"],
            "start_codon_tx_index": start_codon_index,
            "start_codon_genomic_1based": int(obs_coords[start_codon_index]) + 1,
            "start_codon_reference": start_codon_source,
            "utr5_start_exclusion_bp": utr5_start_exclusion_bp,
            "expression_bin": "zero_observed",
        }
        for region, metrics in region_metrics.items():
            add_region_metrics_to_row(row, region, metrics)
            codon_metrics = transcript_codon_metrics(
                obs_values,
                pred_values,
                region_masks[region],
                start_codon_index,
            )
            add_codon_metrics_to_row(row, region, codon_metrics)
            if region in START_CODON_FRAME_REGIONS:
                start_codon_frame_row[f"{region}_bp"] = metrics["bp"]
                add_start_codon_frame_fields(
                    start_codon_frame_row,
                    f"obs_{region}",
                    frame_sums(obs_values, region_masks[region], start_codon_index),
                )
                add_start_codon_frame_fields(
                    start_codon_frame_row,
                    f"pred_{region}",
                    frame_sums(pred_values, region_masks[region], start_codon_index),
                )
            if codon_metrics["n"] > 0:
                region_codon_payloads.append(
                    (
                        row,
                        region,
                        codon_metrics["obs"],
                        codon_metrics["pred"],
                    )
                )
        for frame in range(3):
            row[f"obs_frame{frame}_sum"] = float(obs_frame_sums[frame])
            row[f"obs_frame{frame}_fraction"] = frame_fraction(obs_frame_sums, frame)
            row[f"pred_frame{frame}_sum"] = float(pred_frame_sums[frame])
            row[f"pred_frame{frame}_fraction"] = frame_fraction(pred_frame_sums, frame)
        phase_rows.append(row)
        start_codon_frame_rows.append(start_codon_frame_row)
        transcript_payloads.append((row, obs_values, pred_values, cds_mask, utr5_mask, utr3_mask))
        for region, mask in region_masks.items():
            if region_metrics[region]["bp"] > 0:
                region_payloads.append((row, region, obs_values[mask], pred_values[mask]))

    expressed_means = [row["obs_cds_mean"] for row in phase_rows if row["obs_cds_sum"] > 0]
    if expressed_means:
        lower, upper = np.quantile(expressed_means, [1 / 3, 2 / 3])
    else:
        lower, upper = float("nan"), float("nan")

    for row, *_unused in transcript_payloads:
        if row["obs_cds_sum"] <= 0:
            continue
        bucket = quantile_expression_bin(row["obs_cds_mean"], lower, upper)
        row["expression_bin"] = bucket

    for row in phase_rows:
        row.setdefault("expression_bin", "zero_observed")
    for row, start_codon_frame_row in zip(phase_rows, start_codon_frame_rows):
        start_codon_frame_row["expression_bin"] = row["expression_bin"]

    add_expression_units(phase_rows)

    region_correlation_accs = defaultdict(PositionCorrelationAccumulator)
    region_transcript_counts = defaultdict(int)
    for row, region, obs_region_values, pred_region_values in region_payloads:
        keys = [
            (region, "all", "all"),
            (region, row["expression_bin"], "all"),
            (region, "all", row["strand"]),
            (region, row["expression_bin"], row["strand"]),
        ]
        for key in keys:
            region_correlation_accs[key].add(obs_region_values, pred_region_values)
            region_transcript_counts[key] += 1

    region_correlation_rows = []
    for region in CORRELATION_REGIONS:
        for expression_bin in ("all", "zero_observed", "low", "medium", "high"):
            for strand in ("all", "+", "-"):
                key = (region, expression_bin, strand)
                if key not in region_correlation_accs:
                    continue
                row = {
                    "category": region,
                    "expression_bin": expression_bin,
                    "strand": strand,
                    "transcripts": region_transcript_counts[key],
                }
                row.update(region_correlation_accs[key].summarize())
                region_correlation_rows.append(row)

    cds_correlation_rows = [
        {key: value for key, value in row.items() if key != "category"}
        for row in region_correlation_rows
        if row["category"] == "cds"
    ]

    region_codon_correlation_accs = defaultdict(PositionCorrelationAccumulator)
    region_codon_transcript_counts = defaultdict(int)
    for row, region, obs_codons, pred_codons in region_codon_payloads:
        keys = [
            (region, "all", "all"),
            (region, row["expression_bin"], "all"),
            (region, "all", row["strand"]),
            (region, row["expression_bin"], row["strand"]),
        ]
        for key in keys:
            region_codon_correlation_accs[key].add(obs_codons, pred_codons)
            region_codon_transcript_counts[key] += 1

    region_codon_correlation_rows = []
    for region in CORRELATION_REGIONS:
        for expression_bin in ("all", "zero_observed", "low", "medium", "high"):
            for strand in ("all", "+", "-"):
                key = (region, expression_bin, strand)
                if key not in region_codon_correlation_accs:
                    continue
                row = {
                    "category": region,
                    "expression_bin": expression_bin,
                    "strand": strand,
                    "transcripts": region_codon_transcript_counts[key],
                }
                summary = region_codon_correlation_accs[key].summarize()
                row.update({
                    "n_codons": summary["n_bases"],
                    "obs_mean": summary["obs_mean"],
                    "pred_mean": summary["pred_mean"],
                    "pearson": summary["pearson"],
                    "log1p_pearson": summary["log1p_pearson"],
                })
                region_codon_correlation_rows.append(row)

    cds_codon_correlation_rows = [
        {key: value for key, value in row.items() if key != "category"}
        for row in region_codon_correlation_rows
        if row["category"] == "cds"
    ]

    summary_rows = []
    for bucket in ("all", "zero_observed", "low", "medium", "high"):
        if bucket == "all":
            rows = phase_rows
        else:
            rows = [row for row in phase_rows if row["expression_bin"] == bucket]
        if not rows:
            continue
        valid_match = [int(row["phase_match"]) for row in rows if row["phase_match"] != ""]
        summary_rows.append({
            "expression_bin": bucket,
            "transcripts": len(rows),
            "phase_callable_transcripts": len(valid_match),
            "phase_match_fraction": safe_div(sum(valid_match), len(valid_match)),
            "obs_mean_cds_sum": safe_div(sum(row["obs_cds_sum"] for row in rows), len(rows)),
            "pred_mean_cds_sum": safe_div(sum(row["pred_cds_sum"] for row in rows), len(rows)),
            "obs_mean_frame0_fraction": finite_mean(
                metric_value(row, "obs_frame0_fraction") for row in rows
            ),
            "pred_mean_frame0_fraction": finite_mean(
                metric_value(row, "pred_frame0_fraction") for row in rows
            ),
        })

    ranking_rows = list(phase_rows)
    topk_rows = topk_overlap_rows(ranking_rows)
    start_codon_frame_summary_rows = summarize_start_codon_frame_rows(start_codon_frame_rows)
    for row in summary_rows:
        row["noncoding_exon_transcripts_seen"] = noncoding_count

    return (
        phase_rows,
        summary_rows,
        start_codon_frame_rows,
        start_codon_frame_summary_rows,
        ranking_rows,
        topk_rows,
        cds_correlation_rows,
        region_correlation_rows,
        cds_codon_correlation_rows,
        region_codon_correlation_rows,
    )


def add_boundary_ratio_fields(row, region, obs_ratio, pred_ratio, dropoff):
    row[f"obs_{region}_to_cds_ratio"] = obs_ratio
    row[f"pred_{region}_to_cds_ratio"] = pred_ratio
    ratio_error = (
        pred_ratio - obs_ratio
        if not math.isnan(pred_ratio) and not math.isnan(obs_ratio)
        else float("nan")
    )
    if dropoff:
        row["obs_dropoff"] = (
            1.0 - obs_ratio if not math.isnan(obs_ratio) else float("nan")
        )
        row["pred_dropoff"] = (
            1.0 - pred_ratio if not math.isnan(pred_ratio) else float("nan")
        )
        row["dropoff_error"] = ratio_error
    else:
        row["ratio_error"] = ratio_error


def evaluate_gtf_boundary(
    gtf_path,
    regions,
    observed_plus,
    observed_minus,
    predicted_plus,
    predicted_minus,
    min_cds_bp,
    min_region_bp,
    profile_upstream,
    profile_downstream,
    region,
    region_intervals,
    boundary_index,
    profile_offset_field,
    progress_description,
    dropoff=False,
    transcripts=None,
    grouped_regions=None,
):
    if grouped_regions is None:
        grouped_regions = regions_by_chrom(regions)
    if transcripts is None:
        transcripts = load_gtf_transcripts(gtf_path, grouped_regions)

    transcript_rows = []
    summary_accs = {
        "all": defaultdict(float),
        "+": defaultdict(float),
        "-": defaultdict(float),
    }
    profile_sums = {
        (strand, source): np.zeros(
            profile_upstream + profile_downstream,
            dtype=np.float64,
        )
        for strand in ("all", "+", "-")
        for source in ("observed", "predicted")
    }
    profile_counts = {
        (strand, source): np.zeros(
            profile_upstream + profile_downstream,
            dtype=np.int64,
        )
        for strand in ("all", "+", "-")
        for source in ("observed", "predicted")
    }

    for transcript in tqdm(transcripts, desc=progress_description):
        chrom = transcript["chrom"]
        strand = transcript["strand"]
        obs_bw = observed_plus if strand == "+" else observed_minus
        pred_bw = predicted_plus if strand == "+" else predicted_minus

        cds_intervals = clip_to_regions(chrom, transcript["cds"], grouped_regions)
        region_intervals_clipped = clip_to_regions(
            chrom,
            region_intervals(transcript),
            grouped_regions,
        )
        cds_bp = interval_length(cds_intervals)
        region_bp = interval_length(region_intervals_clipped)
        if cds_bp < min_cds_bp or region_bp < min_region_bp:
            continue

        obs_cds_sum, _ = interval_sum(obs_bw, chrom, cds_intervals)
        pred_cds_sum, _ = interval_sum(pred_bw, chrom, cds_intervals)
        obs_region_sum, _ = interval_sum(
            obs_bw,
            chrom,
            region_intervals_clipped,
        )
        pred_region_sum, _ = interval_sum(
            pred_bw,
            chrom,
            region_intervals_clipped,
        )

        obs_cds_mean = safe_div(obs_cds_sum, cds_bp)
        pred_cds_mean = safe_div(pred_cds_sum, cds_bp)
        obs_region_mean = safe_div(obs_region_sum, region_bp)
        pred_region_mean = safe_div(pred_region_sum, region_bp)
        obs_ratio = safe_ratio(obs_region_mean, obs_cds_mean)
        pred_ratio = safe_ratio(pred_region_mean, pred_cds_mean)

        row = {
            "chrom": chrom,
            "gene_id": transcript["gene_id"],
            "gene_name": transcript["gene_name"],
            "transcript_id": transcript["transcript_id"],
            "transcript_name": transcript["transcript_name"],
            "strand": strand,
            "transcript_start": transcript["transcript_start"],
            "transcript_end": transcript["transcript_end"],
            "cds_bp": cds_bp,
            f"{region}_bp": region_bp,
            "obs_cds_mean": obs_cds_mean,
            "pred_cds_mean": pred_cds_mean,
            f"obs_{region}_mean": obs_region_mean,
            f"pred_{region}_mean": pred_region_mean,
        }
        add_boundary_ratio_fields(row, region, obs_ratio, pred_ratio, dropoff)
        transcript_rows.append(row)

        for key in ("all", strand):
            acc = summary_accs[key]
            acc["transcripts"] += 1
            acc["cds_bp"] += cds_bp
            acc[f"{region}_bp"] += region_bp
            acc["obs_cds_sum"] += obs_cds_sum
            acc["pred_cds_sum"] += pred_cds_sum
            acc[f"obs_{region}_sum"] += obs_region_sum
            acc[f"pred_{region}_sum"] += pred_region_sum

        obs_coords, obs_values = oriented_exon_arrays(transcript, obs_bw, grouped_regions)
        pred_coords, pred_values = oriented_exon_arrays(transcript, pred_bw, grouped_regions)
        boundary = boundary_index(transcript, obs_coords)
        if boundary is not None and len(obs_coords) == len(pred_coords):
            for source, values in (("observed", obs_values), ("predicted", pred_values)):
                profile = profile_slice(
                    values,
                    boundary,
                    profile_upstream,
                    profile_downstream,
                )
                valid = np.isfinite(profile)
                for key in ((strand, source), ("all", source)):
                    profile_sums[key][valid] += profile[valid]
                    profile_counts[key][valid] += 1

    summary_rows = []
    for strand, acc in summary_accs.items():
        transcripts_count = int(acc["transcripts"])
        obs_cds_mean = safe_div(acc["obs_cds_sum"], acc["cds_bp"])
        pred_cds_mean = safe_div(acc["pred_cds_sum"], acc["cds_bp"])
        obs_region_mean = safe_div(
            acc[f"obs_{region}_sum"],
            acc[f"{region}_bp"],
        )
        pred_region_mean = safe_div(
            acc[f"pred_{region}_sum"],
            acc[f"{region}_bp"],
        )
        obs_ratio = safe_ratio(obs_region_mean, obs_cds_mean)
        pred_ratio = safe_ratio(pred_region_mean, pred_cds_mean)
        row = {
            "strand": strand,
            "transcripts": transcripts_count,
            "cds_bp": int(acc["cds_bp"]),
            f"{region}_bp": int(acc[f"{region}_bp"]),
            "obs_cds_mean": obs_cds_mean,
            "pred_cds_mean": pred_cds_mean,
            f"obs_{region}_mean": obs_region_mean,
            f"pred_{region}_mean": pred_region_mean,
        }
        add_boundary_ratio_fields(row, region, obs_ratio, pred_ratio, dropoff)
        summary_rows.append(row)

    profile_rows = []
    for strand in ("all", "+", "-"):
        for source in ("observed", "predicted"):
            sums = profile_sums[(strand, source)]
            counts = profile_counts[(strand, source)]
            means = np.divide(
                sums,
                counts,
                out=np.full_like(sums, np.nan, dtype=np.float64),
                where=counts > 0,
            )
            for index, mean in enumerate(means):
                profile_rows.append({
                    "strand": strand,
                    "source": source,
                    profile_offset_field: index - profile_upstream,
                    "mean_signal": mean,
                    "n": int(counts[index]),
                })

    return transcript_rows, summary_rows, profile_rows


def evaluate_gtf_dropoff(
    gtf_path,
    regions,
    observed_plus,
    observed_minus,
    predicted_plus,
    predicted_minus,
    min_cds_bp,
    min_utr3_bp,
    stop_upstream,
    stop_downstream,
    transcripts=None,
    grouped_regions=None,
):
    return evaluate_gtf_boundary(
        gtf_path,
        regions,
        observed_plus,
        observed_minus,
        predicted_plus,
        predicted_minus,
        min_cds_bp,
        min_utr3_bp,
        stop_upstream,
        stop_downstream,
        "utr3",
        utr3_intervals,
        stop_boundary_index,
        "offset_from_after_stop",
        "Evaluating GTF CDS/3'UTR",
        dropoff=True,
        transcripts=transcripts,
        grouped_regions=grouped_regions,
    )


def evaluate_gtf_start_boundary(
    gtf_path,
    regions,
    observed_plus,
    observed_minus,
    predicted_plus,
    predicted_minus,
    min_cds_bp,
    min_utr5_bp,
    start_upstream,
    start_downstream,
    transcripts=None,
    grouped_regions=None,
):
    return evaluate_gtf_boundary(
        gtf_path,
        regions,
        observed_plus,
        observed_minus,
        predicted_plus,
        predicted_minus,
        min_cds_bp,
        min_utr5_bp,
        start_upstream,
        start_downstream,
        "utr5",
        utr5_intervals,
        start_boundary_index,
        "offset_from_cds_start",
        "Evaluating GTF 5'UTR/CDS",
        transcripts=transcripts,
        grouped_regions=grouped_regions,
    )


def metric_value(row, field):
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def write_plots(
    plot_dir,
    out_prefix,
    metric_rows,
    gtf_summary_rows,
    stop_profile_rows,
    gtf_phase_summary_rows=None,
    start_codon_frame_summary_rows=None,
    topk_rows=None,
    ranking_rows=None,
    cds_correlation_rows=None,
    filtered_region_transcript_r_rows=None,
    filtered_codon_transcript_r_rows=None,
    utr5_start_exclusion_bp=30,
    gtf_start_summary_rows=None,
    start_profile_rows=None,
):
    gtf_phase_summary_rows = gtf_phase_summary_rows or []
    start_codon_frame_summary_rows = start_codon_frame_summary_rows or []
    topk_rows = topk_rows or []
    ranking_rows = ranking_rows or []
    cds_correlation_rows = cds_correlation_rows or []
    filtered_region_transcript_r_rows = filtered_region_transcript_r_rows or []
    filtered_codon_transcript_r_rows = filtered_codon_transcript_r_rows or []
    gtf_start_summary_rows = gtf_start_summary_rows or []
    start_profile_rows = start_profile_rows or []

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = plot_dir / ".matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = Path(out_prefix).name

    def plot_boundary(
        summary_rows,
        profile_rows,
        region,
        ratio_ylabel,
        ratio_title,
        summary_suffix,
        offset_field,
        profile_xlabel,
        profile_title,
        profile_suffix,
    ):
        if summary_rows:
            strands = [row["strand"] for row in summary_rows]
            x = np.arange(len(strands))
            width = 0.35
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(12, 4),
                constrained_layout=True,
            )
            for offset, source in (
                (-width / 2, "observed"),
                (width / 2, "predicted"),
            ):
                prefix_key = "obs" if source == "observed" else "pred"
                axes[0].bar(
                    x + offset,
                    [
                        metric_value(
                            row,
                            f"{prefix_key}_{region}_to_cds_ratio",
                        )
                        for row in summary_rows
                    ],
                    width,
                    label=source,
                )
                axes[1].bar(
                    x + offset,
                    [
                        metric_value(row, f"{prefix_key}_cds_mean")
                        for row in summary_rows
                    ],
                    width,
                    label=f"{source} CDS",
                )
            for ax in axes:
                ax.set_xticks(x)
                ax.set_xticklabels(strands)
                ax.legend()
            axes[0].set_ylabel(ratio_ylabel)
            axes[0].set_title(ratio_title)
            axes[1].set_ylabel("mean signal")
            axes[1].set_title("CDS Calibration")
            fig.savefig(plot_dir / f"{prefix}.{summary_suffix}.png", dpi=200)
            plt.close(fig)

        if profile_rows:
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            for source in ("observed", "predicted"):
                rows = [
                    row for row in profile_rows
                    if row["strand"] == "all" and row["source"] == source
                ]
                rows.sort(key=lambda row: int(row[offset_field]))
                ax.plot(
                    [int(row[offset_field]) for row in rows],
                    [metric_value(row, "mean_signal") for row in rows],
                    label=source,
                )
            ax.axvline(0, color="black", linewidth=1, alpha=0.6)
            ax.set_xlabel(profile_xlabel)
            ax.set_ylabel("mean signal")
            ax.set_title(profile_title)
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.savefig(plot_dir / f"{prefix}.{profile_suffix}.png", dpi=200)
            plt.close(fig)

    comparisons = ["plus", "minus", "combined", "pred_plus_vs_obs_minus", "pred_minus_vs_obs_plus"]
    labels = {
        "plus": "plus",
        "minus": "minus",
        "combined": "combined",
        "pred_plus_vs_obs_minus": "plus vs obs minus",
        "pred_minus_vs_obs_plus": "minus vs obs plus",
    }
    rows_by_comparison = defaultdict(list)
    for row in metric_rows:
        rows_by_comparison[row["comparison"]].append(row)
    for rows in rows_by_comparison.values():
        rows.sort(key=lambda row: int(row["bin_size"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for comparison in comparisons:
        rows = rows_by_comparison.get(comparison, [])
        if not rows:
            continue
        x = [int(row["bin_size"]) for row in rows]
        axes[0].plot(x, [metric_value(row, "log1p_pearson") for row in rows], marker="o", label=labels[comparison])
        axes[1].plot(x, [metric_value(row, "auprc_nonzero") for row in rows], marker="o", label=labels[comparison])
    for ax, title, ylabel in (
        (axes[0], "Correlation By Bin Size", "log1p Pearson"),
        (axes[1], "Nonzero Detection By Bin Size", "AUPRC"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel("bin size (bp)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(plot_dir / f"{prefix}.metrics.png", dpi=200)
    plt.close(fig)

    combined_rows = rows_by_comparison.get("combined", [])
    if combined_rows:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        x = np.arange(len(combined_rows))
        width = 0.35
        ax.bar(x - width / 2, [metric_value(row, "obs_mean") for row in combined_rows], width, label="observed")
        ax.bar(x + width / 2, [metric_value(row, "pred_mean") for row in combined_rows], width, label="predicted")
        ax.set_xticks(x)
        ax.set_xticklabels([row["bin_size"] for row in combined_rows])
        ax.set_xlabel("bin size (bp)")
        ax.set_ylabel("mean binned signal")
        ax.set_title("Combined Calibration")
        ax.legend()
        fig.savefig(plot_dir / f"{prefix}.calibration.png", dpi=200)
        plt.close(fig)

    plot_boundary(
        gtf_summary_rows,
        stop_profile_rows,
        "utr3",
        "3'UTR / CDS mean signal",
        "Dropoff Ratio After Stop",
        "gtf_dropoff",
        "offset_from_after_stop",
        "transcript-space offset from first base after stop codon",
        "Stop Codon Metaprofile",
        "stop_profile",
    )
    plot_boundary(
        gtf_start_summary_rows,
        start_profile_rows,
        "utr5",
        "5'UTR / CDS mean signal",
        "5'UTR/CDS Ratio At Start",
        "gtf_start_boundary",
        "offset_from_cds_start",
        "transcript-space offset from first CDS base",
        "5'UTR/CDS Boundary Metaprofile",
        "start_profile",
    )

    if gtf_phase_summary_rows:
        rows = [
            row for row in gtf_phase_summary_rows
            if row["expression_bin"] in {"zero_observed", "low", "medium", "high", "all"}
        ]
        order = {"zero_observed": 0, "low": 1, "medium": 2, "high": 3, "all": 4}
        rows.sort(key=lambda row: order.get(row["expression_bin"], 99))
        labels = [row["expression_bin"] for row in rows]
        x = np.arange(len(rows))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        axes[0].bar(x, [metric_value(row, "phase_match_fraction") for row in rows])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=30, ha="right")
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("observed/predicted dominant-frame match")
        axes[0].set_title("CDS Phase Agreement")

        width = 0.35
        axes[1].bar(
            x - width / 2,
            [metric_value(row, "obs_mean_frame0_fraction") for row in rows],
            width,
            label="observed",
        )
        axes[1].bar(
            x + width / 2,
            [metric_value(row, "pred_mean_frame0_fraction") for row in rows],
            width,
            label="predicted",
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=30, ha="right")
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("mean frame-0 fraction")
        axes[1].set_title("Frame-0 Bias By Expression")
        axes[1].legend()
        fig.savefig(plot_dir / f"{prefix}.gtf_phase_agreement.png", dpi=200)
        plt.close(fig)

    if start_codon_frame_summary_rows:
        rows = [
            row for row in start_codon_frame_summary_rows
            if row["expression_bin"] == "all" and row["strand"] == "all"
        ]
        by_category_source = {
            (row["category"], row["source"]): row
            for row in rows
        }

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True, sharey=True)
        x = np.arange(3)
        width = 0.35
        titles = {
            "utr5": "5'UTR",
            "cds": "CDS",
            "utr3": "3'UTR",
        }
        for ax, category in zip(axes, START_CODON_FRAME_REGIONS):
            for offset, source in ((-width / 2, "observed"), (width / 2, "predicted")):
                row = by_category_source.get((category, source))
                if row is None:
                    continue
                ax.bar(
                    x + offset,
                    [
                        metric_value(row, f"pooled_frame{frame}_fraction")
                        for frame in range(3)
                    ],
                    width,
                    label=source,
                )
            ax.axhline(1 / 3, color="black", linestyle=":", linewidth=1, alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(["0", "1", "2"])
            ax.set_xlabel("frame relative to annotated start codon")
            ax.set_title(titles[category])
            ax.grid(True, axis="y", alpha=0.25)
        axes[0].set_ylabel("pooled signal fraction")
        axes[-1].legend(fontsize=8)
        fig.suptitle("Start-Codon-Aligned Frame Periodicity")
        fig.savefig(plot_dir / f"{prefix}.gtf_start_codon_frame_periodicity.png", dpi=200)
        plt.close(fig)

    if topk_rows:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        rows = sorted(topk_rows, key=lambda row: (row["metric"], int(row["requested_top_n"])))
        x = np.arange(len(rows))
        metric_labels = {
            "cds_signal_ppm": "CDS signal ppm",
            "cds_signal_per_kb_ppm": "CDS signal/kb ppm",
        }
        labels = [
            f"{metric_labels.get(row['metric'], row['metric'])} top{row['requested_top_n']}"
            for row in rows
        ]
        ax.bar(x, [metric_value(row, "overlap_fraction") for row in rows])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("observed/predicted top-k overlap")
        ax.set_title("Top Translated CDS Ranking Agreement")
        fig.savefig(plot_dir / f"{prefix}.gtf_topk_overlap.png", dpi=200)
        plt.close(fig)

    if ranking_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        ranking_metrics = (
            ("cds_signal_ppm", "CDS signal ppm"),
            ("cds_signal_per_kb_ppm", "CDS signal/kb ppm"),
        )
        for ax, (metric, label) in zip(axes, ranking_metrics):
            obs = np.asarray([metric_value(row, f"obs_{metric}") for row in ranking_rows], dtype=np.float64)
            pred = np.asarray([metric_value(row, f"pred_{metric}") for row in ranking_rows], dtype=np.float64)
            valid = np.isfinite(obs) & np.isfinite(pred)
            ax.scatter(np.log1p(obs[valid]), np.log1p(pred[valid]), s=8, alpha=0.45)
            corr = pearson_from_sums(
                int(valid.sum()),
                float(np.log1p(obs[valid]).sum()),
                float(np.log1p(pred[valid]).sum()),
                float(np.square(np.log1p(obs[valid])).sum()),
                float(np.square(np.log1p(pred[valid])).sum()),
                float((np.log1p(obs[valid]) * np.log1p(pred[valid])).sum()),
            )
            ax.set_xlabel(f"observed log1p {label}")
            ax.set_ylabel(f"predicted log1p {label}")
            ax.set_title(f"{label} ranking signal, r={corr:.3f}")
            ax.grid(True, alpha=0.25)
        fig.savefig(plot_dir / f"{prefix}.gtf_translation_scatter.png", dpi=200)
        plt.close(fig)

    if cds_correlation_rows:
        rows = [
            row for row in cds_correlation_rows
            if row["strand"] == "all"
            and row["expression_bin"] in {"all", "zero_observed", "low", "medium", "high"}
        ]
        order = {"zero_observed": 0, "low": 1, "medium": 2, "high": 3, "all": 4}
        rows.sort(key=lambda row: order.get(row["expression_bin"], 99))
        labels = [row["expression_bin"] for row in rows]
        x = np.arange(len(rows))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].bar(x, [metric_value(row, "pearson") for row in rows])
        axes[1].bar(x, [metric_value(row, "log1p_pearson") for row in rows])
        for ax, title in (
            (axes[0], "Raw CDS-Position Pearson"),
            (axes[1], "log1p CDS-Position Pearson"),
        ):
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("Pearson")
            ax.set_title(title)
            ax.grid(True, axis="y", alpha=0.25)
        fig.savefig(plot_dir / f"{prefix}.gtf_cds_position_correlations.png", dpi=200)
        plt.close(fig)

    fisher_datasets = (
        ("Nucleotide", filtered_region_transcript_r_rows),
        ("Codon", filtered_codon_transcript_r_rows),
    )
    if any(rows for _label, rows in fisher_datasets):
        categories = CORRELATION_REGIONS
        category_titles = {
            "cds": "CDS",
            "utr5": "5'UTR",
            "utr5_distal": f"5'UTR excluding last {utr5_start_exclusion_bp} nt",
            "utr3": "3'UTR",
        }
        expression_order = ("low", "medium", "high", "all")
        fig, axes = plt.subplots(
            2,
            len(categories),
            figsize=(17, 7),
            constrained_layout=True,
            sharey=True,
        )
        width = 0.36

        for row_index, (resolution, summary_rows) in enumerate(fisher_datasets):
            for column_index, category in enumerate(categories):
                ax = axes[row_index, column_index]
                rows = {
                    row["expression_bin"]: row
                    for row in summary_rows
                    if row["category"] == category
                    and row["strand"] == "all"
                    and row["expression_bin"] in expression_order
                }
                labels = [bucket for bucket in expression_order if bucket in rows]
                x = np.arange(len(labels))
                ax.bar(
                    x - width / 2,
                    [metric_value(rows[bucket], "fisher_mean_r") for bucket in labels],
                    width,
                    label="raw",
                )
                ax.bar(
                    x + width / 2,
                    [
                        metric_value(rows[bucket], "fisher_mean_log1p_r")
                        for bucket in labels
                    ],
                    width,
                    label="log1p",
                )
                ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
                ax.set_ylim(-1, 1)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha="right")
                ax.set_title(f"{resolution}: {category_titles[category]}")
                ax.grid(True, axis="y", alpha=0.25)
                if column_index == 0:
                    ax.set_ylabel("Fisher-mean per-transcript Pearson r")
                if row_index == 1:
                    ax.set_xlabel("observed-expression bin")
        axes[0, -1].legend(fontsize=8)
        fig.suptitle("Coverage-Filtered Transcript Shape Correlations")
        fig.savefig(
            plot_dir / f"{prefix}.gtf_filtered_per_transcript_fisher_correlations.png",
            dpi=200,
        )
        plt.close(fig)

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10, 7),
            constrained_layout=True,
            sharey=True,
        )
        width = 0.36
        for row_index, (resolution, summary_rows) in enumerate(fisher_datasets):
            for column_index, (metric, metric_label) in enumerate(
                (("fisher_mean_r", "raw"), ("fisher_mean_log1p_r", "log1p"))
            ):
                ax = axes[row_index, column_index]
                rows = {
                    (row["category"], row["expression_bin"]): row
                    for row in summary_rows
                    if row["category"] in {"utr5", "utr5_distal"}
                    and row["strand"] == "all"
                    and row["expression_bin"] in expression_order
                }
                labels = [
                    bucket
                    for bucket in expression_order
                    if ("utr5", bucket) in rows
                    or ("utr5_distal", bucket) in rows
                ]
                x = np.arange(len(labels))
                ax.bar(
                    x - width / 2,
                    [
                        metric_value(rows.get(("utr5", bucket), {}), metric)
                        for bucket in labels
                    ],
                    width,
                    label="full 5'UTR",
                )
                ax.bar(
                    x + width / 2,
                    [
                        metric_value(rows.get(("utr5_distal", bucket), {}), metric)
                        for bucket in labels
                    ],
                    width,
                    label=f"exclude last {utr5_start_exclusion_bp} nt",
                )
                ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
                ax.set_ylim(-1, 1)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha="right")
                ax.set_title(f"{resolution}, {metric_label}")
                ax.grid(True, axis="y", alpha=0.25)
                if column_index == 0:
                    ax.set_ylabel("Fisher-mean per-transcript Pearson r")
                if row_index == 1:
                    ax.set_xlabel("observed-expression bin")
        axes[0, -1].legend(fontsize=8)
        fig.suptitle("5'UTR Shape With Start-Proximal Segment Removed")
        fig.savefig(
            plot_dir / f"{prefix}.gtf_utr5_start_exclusion_fisher_correlations.png",
            dpi=200,
        )
        plt.close(fig)


def main():
    args = parse_args()
    if args.start_upstream < 0 or args.start_downstream < 0:
        raise ValueError("--start-upstream and --start-downstream must be non-negative")
    if args.start_upstream + args.start_downstream == 0:
        raise ValueError("Start-boundary metaprofile window must contain at least one base")
    if args.min_utr5_bp < 0:
        raise ValueError("--min-utr5-bp must be non-negative")

    bin_sizes = parse_ints(args.bin_sizes)
    regions = load_regions(args.regions_bed)
    chroms = parse_chroms(args.chroms)
    if chroms:
        original_region_count = len(regions)
        regions = filter_regions_by_chroms(regions, chroms)
        print(
            "Filtered regions to chromosomes "
            f"{','.join(chroms)} "
            f"({len(regions)} of {original_region_count} intervals)"
        )

    observed_plus = BigWigReplicates(args.observed_plus)
    observed_minus = BigWigReplicates(args.observed_minus)
    predicted_plus = pyBigWig.open(args.predicted_plus)
    predicted_minus = pyBigWig.open(args.predicted_minus)

    metric_rows = []
    gtf_transcript_rows = []
    gtf_summary_rows = []
    stop_profile_rows = []
    gtf_start_transcript_rows = []
    gtf_start_summary_rows = []
    start_profile_rows = []
    gtf_phase_rows = []
    gtf_phase_summary_rows = []
    start_codon_frame_rows = []
    start_codon_frame_summary_rows = []
    ranking_rows = []
    topk_rows = []
    cds_correlation_rows = []
    region_correlation_rows = []
    cds_codon_correlation_rows = []
    region_codon_correlation_rows = []
    filtered_transcript_r_rows = []
    filtered_region_transcript_r_rows = []
    filtered_codon_transcript_r_rows = []
    try:
        for bin_size in bin_sizes:
            accs = make_accumulators(
                bin_size,
                regions,
                args.max_rank_points,
                args.positive_threshold,
            )

            for chrom, start, end in tqdm(regions, desc=f"Evaluating {bin_size} bp bins"):
                obs_plus = bw_values(observed_plus, chrom, start, end)
                obs_minus = bw_values(observed_minus, chrom, start, end)
                pred_plus = bw_values(predicted_plus, chrom, start, end)
                pred_minus = bw_values(predicted_minus, chrom, start, end)

                obs_plus_binned = bin_values(obs_plus, bin_size)
                obs_minus_binned = bin_values(obs_minus, bin_size)
                pred_plus_binned = bin_values(pred_plus, bin_size)
                pred_minus_binned = bin_values(pred_minus, bin_size)

                accs["plus"].add(obs_plus_binned, pred_plus_binned)
                accs["minus"].add(obs_minus_binned, pred_minus_binned)
                accs["combined"].add(
                    obs_plus_binned + obs_minus_binned,
                    pred_plus_binned + pred_minus_binned,
                )
                accs["pred_plus_vs_obs_minus"].add(obs_minus_binned, pred_plus_binned)
                accs["pred_minus_vs_obs_plus"].add(obs_plus_binned, pred_minus_binned)
                accs["pred_plus_vs_pred_minus"].add(pred_minus_binned, pred_plus_binned)

            for comparison, acc in accs.items():
                row = {"bin_size": bin_size, "comparison": comparison}
                row.update(acc.summarize())
                metric_rows.append(row)

        if args.gtf and args.gtf.lower() not in {"none", "null", "false", "0"}:
            if Path(args.gtf).exists():
                grouped_regions = regions_by_chrom(regions)
                transcripts = load_gtf_transcripts(args.gtf, grouped_regions)
                gtf_transcript_rows, gtf_summary_rows, stop_profile_rows = evaluate_gtf_dropoff(
                    args.gtf,
                    regions,
                    observed_plus,
                    observed_minus,
                    predicted_plus,
                    predicted_minus,
                    args.min_cds_bp,
                    args.min_utr3_bp,
                    args.stop_upstream,
                    args.stop_downstream,
                    transcripts=transcripts,
                    grouped_regions=grouped_regions,
                )
                (
                    gtf_start_transcript_rows,
                    gtf_start_summary_rows,
                    start_profile_rows,
                ) = evaluate_gtf_start_boundary(
                    args.gtf,
                    regions,
                    observed_plus,
                    observed_minus,
                    predicted_plus,
                    predicted_minus,
                    args.min_cds_bp,
                    args.min_utr5_bp,
                    args.start_upstream,
                    args.start_downstream,
                    transcripts=transcripts,
                    grouped_regions=grouped_regions,
                )
                (
                    gtf_phase_rows,
                    gtf_phase_summary_rows,
                    start_codon_frame_rows,
                    start_codon_frame_summary_rows,
                    ranking_rows,
                    topk_rows,
                    cds_correlation_rows,
                    region_correlation_rows,
                    cds_codon_correlation_rows,
                    region_codon_correlation_rows,
                ) = evaluate_phase_controls(
                    transcripts,
                    grouped_regions,
                    observed_plus,
                    observed_minus,
                    predicted_plus,
                    predicted_minus,
                    args.min_cds_bp,
                    args.utr5_start_exclusion_bp,
                )
                filtered_transcript_r_rows = filtered_per_transcript_r_rows(
                    gtf_phase_rows,
                    args.filtered_transcript_min_obs_mean,
                )
                filtered_region_transcript_r_rows = filtered_per_transcript_region_r_rows(
                    gtf_phase_rows,
                    args.filtered_transcript_min_obs_mean,
                )
                filtered_codon_transcript_r_rows = filtered_per_transcript_codon_r_rows(
                    gtf_phase_rows,
                    args.filtered_transcript_min_obs_mean,
                )
            else:
                print(f"Skipping GTF evaluation because file does not exist: {args.gtf}")
    finally:
        observed_plus.close()
        observed_minus.close()
        predicted_plus.close()
        predicted_minus.close()

    metric_fields = [
        "bin_size",
        "comparison",
        "n",
        "obs_mean",
        "pred_mean",
        "obs_sum",
        "pred_sum",
        "pearson",
        "log1p_pearson",
        "spearman",
        "mae",
        "rmse",
        "poisson_nll_no_constant",
        "positive_fraction",
        "auroc_nonzero",
        "auprc_nonzero",
        "auprc_baseline",
        "auprc_lift",
        "max_f1_nonzero",
        "max_f1_threshold",
        "top_1pct_precision_nonzero",
        "top_1pct_recall_nonzero",
        "top_1pct_enrichment_nonzero",
        "top_5pct_precision_nonzero",
        "top_5pct_recall_nonzero",
        "top_5pct_enrichment_nonzero",
        "top_10pct_precision_nonzero",
        "top_10pct_recall_nonzero",
        "top_10pct_enrichment_nonzero",
        "top_1pct_signal_recovery",
        "top_5pct_signal_recovery",
        "top_10pct_signal_recovery",
    ]
    metrics_path = f"{args.out_prefix}.metrics.csv"
    write_csv(metrics_path, metric_rows, metric_fields)

    gtf_transcripts_path = None
    gtf_summary_path = None
    stop_profile_path = None
    gtf_start_transcripts_path = None
    gtf_start_summary_path = None
    start_profile_path = None
    gtf_phase_path = None
    gtf_phase_summary_path = None
    start_codon_frame_path = None
    start_codon_frame_summary_path = None
    ranking_path = None
    topk_path = None
    cds_correlation_path = None
    region_correlation_path = None
    cds_codon_correlation_path = None
    region_codon_correlation_path = None
    filtered_transcript_r_path = None
    filtered_region_transcript_r_path = None
    filtered_codon_transcript_r_path = None
    if gtf_summary_rows:
        gtf_transcripts_path = f"{args.out_prefix}.gtf_transcripts.csv"
        gtf_summary_path = f"{args.out_prefix}.gtf_dropoff_summary.csv"
        stop_profile_path = f"{args.out_prefix}.stop_profile.csv"
        write_csv(
            gtf_transcripts_path,
            gtf_transcript_rows,
            [
                "chrom",
                "gene_id",
                "gene_name",
                "transcript_id",
                "transcript_name",
                "strand",
                "transcript_start",
                "transcript_end",
                "cds_bp",
                "utr3_bp",
                "obs_cds_mean",
                "pred_cds_mean",
                "obs_utr3_mean",
                "pred_utr3_mean",
                "obs_utr3_to_cds_ratio",
                "pred_utr3_to_cds_ratio",
                "obs_dropoff",
                "pred_dropoff",
                "dropoff_error",
            ],
        )
        write_csv(
            gtf_summary_path,
            gtf_summary_rows,
            [
                "strand",
                "transcripts",
                "cds_bp",
                "utr3_bp",
                "obs_cds_mean",
                "pred_cds_mean",
                "obs_utr3_mean",
                "pred_utr3_mean",
                "obs_utr3_to_cds_ratio",
                "pred_utr3_to_cds_ratio",
                "obs_dropoff",
                "pred_dropoff",
                "dropoff_error",
            ],
        )
        write_csv(
            stop_profile_path,
            stop_profile_rows,
            ["strand", "source", "offset_from_after_stop", "mean_signal", "n"],
        )

    if gtf_start_summary_rows:
        gtf_start_transcripts_path = (
            f"{args.out_prefix}.gtf_start_boundary_by_transcript.csv"
        )
        gtf_start_summary_path = f"{args.out_prefix}.gtf_start_boundary_summary.csv"
        start_profile_path = f"{args.out_prefix}.start_profile.csv"
        write_csv(
            gtf_start_transcripts_path,
            gtf_start_transcript_rows,
            [
                "chrom",
                "gene_id",
                "gene_name",
                "transcript_id",
                "transcript_name",
                "strand",
                "transcript_start",
                "transcript_end",
                "cds_bp",
                "utr5_bp",
                "obs_cds_mean",
                "pred_cds_mean",
                "obs_utr5_mean",
                "pred_utr5_mean",
                "obs_utr5_to_cds_ratio",
                "pred_utr5_to_cds_ratio",
                "ratio_error",
            ],
        )
        write_csv(
            gtf_start_summary_path,
            gtf_start_summary_rows,
            [
                "strand",
                "transcripts",
                "cds_bp",
                "utr5_bp",
                "obs_cds_mean",
                "pred_cds_mean",
                "obs_utr5_mean",
                "pred_utr5_mean",
                "obs_utr5_to_cds_ratio",
                "pred_utr5_to_cds_ratio",
                "ratio_error",
            ],
        )
        write_csv(
            start_profile_path,
            start_profile_rows,
            ["strand", "source", "offset_from_cds_start", "mean_signal", "n"],
        )

    if gtf_phase_rows:
        phase_fields = [
            "chrom",
            "gene_id",
            "gene_type",
            "gene_name",
            "transcript_id",
            "transcript_type",
            "transcript_name",
            "strand",
            "transcript_start",
            "transcript_end",
            "cds_bp",
            "expression_bin",
            "obs_cds_sum",
            "pred_cds_sum",
            "obs_cds_mean",
            "pred_cds_mean",
            "cds_position_n",
            "cds_position_pearson",
            "cds_position_log1p_pearson",
            "cds_profile_n",
            "cds_profile_pearson",
            "cds_codon_n",
            "cds_codon_pearson",
            "cds_codon_log1p_pearson",
            "utr5_bp",
            "obs_utr5_sum",
            "pred_utr5_sum",
            "obs_utr5_mean",
            "pred_utr5_mean",
            "utr5_position_n",
            "utr5_position_pearson",
            "utr5_position_log1p_pearson",
            "utr5_codon_n",
            "utr5_codon_pearson",
            "utr5_codon_log1p_pearson",
            "utr5_distal_bp",
            "obs_utr5_distal_sum",
            "pred_utr5_distal_sum",
            "obs_utr5_distal_mean",
            "pred_utr5_distal_mean",
            "utr5_distal_position_n",
            "utr5_distal_position_pearson",
            "utr5_distal_position_log1p_pearson",
            "utr5_distal_codon_n",
            "utr5_distal_codon_pearson",
            "utr5_distal_codon_log1p_pearson",
            "utr3_bp",
            "obs_utr3_sum",
            "pred_utr3_sum",
            "obs_utr3_mean",
            "pred_utr3_mean",
            "utr3_position_n",
            "utr3_position_pearson",
            "utr3_position_log1p_pearson",
            "utr3_codon_n",
            "utr3_codon_pearson",
            "utr3_codon_log1p_pearson",
            "obs_cds_signal_ppm",
            "pred_cds_signal_ppm",
            "obs_cds_signal_per_kb_ppm",
            "pred_cds_signal_per_kb_ppm",
            "obs_cds_signal_ppm_rank",
            "pred_cds_signal_ppm_rank",
            "obs_cds_signal_per_kb_ppm_rank",
            "pred_cds_signal_per_kb_ppm_rank",
            "obs_dominant_frame",
            "pred_dominant_frame",
            "phase_match",
            "obs_frame0_sum",
            "obs_frame0_fraction",
            "obs_frame1_sum",
            "obs_frame1_fraction",
            "obs_frame2_sum",
            "obs_frame2_fraction",
            "pred_frame0_sum",
            "pred_frame0_fraction",
            "pred_frame1_sum",
            "pred_frame1_fraction",
            "pred_frame2_sum",
            "pred_frame2_fraction",
        ]
        gtf_phase_path = f"{args.out_prefix}.gtf_phase_by_transcript.csv"
        ranking_path = f"{args.out_prefix}.gtf_translation_ranking.csv"
        write_csv(gtf_phase_path, gtf_phase_rows, phase_fields)
        write_csv(ranking_path, ranking_rows, phase_fields)

        gtf_phase_summary_path = f"{args.out_prefix}.gtf_phase_summary.csv"
        write_csv(
            gtf_phase_summary_path,
            gtf_phase_summary_rows,
            [
                "expression_bin",
                "transcripts",
                "phase_callable_transcripts",
                "phase_match_fraction",
                "obs_mean_cds_sum",
                "pred_mean_cds_sum",
                "obs_mean_frame0_fraction",
                "pred_mean_frame0_fraction",
                "noncoding_exon_transcripts_seen",
            ],
        )

        start_codon_frame_path = f"{args.out_prefix}.gtf_start_codon_frame_by_transcript.csv"
        write_csv(
            start_codon_frame_path,
            start_codon_frame_rows,
            start_codon_frame_fields(),
        )

        start_codon_frame_summary_path = f"{args.out_prefix}.gtf_start_codon_frame_summary.csv"
        write_csv(
            start_codon_frame_summary_path,
            start_codon_frame_summary_rows,
            start_codon_frame_summary_fields(),
        )

        topk_path = f"{args.out_prefix}.gtf_translation_topk.csv"
        write_csv(
            topk_path,
            topk_rows,
            [
                "metric",
                "requested_top_n",
                "used_top_n",
                "available_transcripts",
                "overlap_count",
                "overlap_fraction",
                "obs_top_genes",
                "pred_top_genes",
                "overlap_genes",
            ],
        )

        cds_correlation_path = f"{args.out_prefix}.gtf_cds_position_correlations.csv"
        write_csv(
            cds_correlation_path,
            cds_correlation_rows,
            [
                "expression_bin",
                "strand",
                "transcripts",
                "n_bases",
                "obs_mean",
                "pred_mean",
                "pearson",
                "log1p_pearson",
            ],
        )

        region_correlation_path = f"{args.out_prefix}.gtf_region_position_correlations.csv"
        write_csv(
            region_correlation_path,
            region_correlation_rows,
            [
                "category",
                "expression_bin",
                "strand",
                "transcripts",
                "n_bases",
                "obs_mean",
                "pred_mean",
                "pearson",
                "log1p_pearson",
            ],
        )

        cds_codon_correlation_path = f"{args.out_prefix}.gtf_cds_codon_correlations.csv"
        write_csv(
            cds_codon_correlation_path,
            cds_codon_correlation_rows,
            [
                "expression_bin",
                "strand",
                "transcripts",
                "n_codons",
                "obs_mean",
                "pred_mean",
                "pearson",
                "log1p_pearson",
            ],
        )

        region_codon_correlation_path = (
            f"{args.out_prefix}.gtf_region_codon_correlations.csv"
        )
        write_csv(
            region_codon_correlation_path,
            region_codon_correlation_rows,
            [
                "category",
                "expression_bin",
                "strand",
                "transcripts",
                "n_codons",
                "obs_mean",
                "pred_mean",
                "pearson",
                "log1p_pearson",
            ],
        )

        filtered_transcript_r_path = f"{args.out_prefix}.gtf_filtered_per_transcript_r.csv"
        write_csv(
            filtered_transcript_r_path,
            filtered_transcript_r_rows,
            [
                "expression_bin",
                "strand",
                "min_obs_cds_mean_exclusive",
                "transcripts",
                "coverage_passing_transcripts",
                "pearson_callable_transcripts",
                "log1p_pearson_callable_transcripts",
                "profile_pearson_callable_transcripts",
                "mean_obs_cds_mean",
                "median_obs_cds_mean",
                "mean_pred_cds_mean",
                "median_pred_cds_mean",
                "mean_r",
                "median_r",
                "fisher_mean_r",
                "cds_bp_weighted_mean_r",
                "mean_log1p_r",
                "median_log1p_r",
                "fisher_mean_log1p_r",
                "cds_bp_weighted_mean_log1p_r",
                "mean_profile_r",
                "median_profile_r",
                "fisher_mean_profile_r",
                "cds_bp_weighted_mean_profile_r",
            ],
        )

        filtered_region_transcript_r_path = (
            f"{args.out_prefix}.gtf_filtered_per_transcript_region_r.csv"
        )
        write_csv(
            filtered_region_transcript_r_path,
            filtered_region_transcript_r_rows,
            [
                "category",
                "expression_bin",
                "strand",
                "min_obs_mean_exclusive",
                "transcripts",
                "coverage_passing_transcripts",
                "pearson_callable_transcripts",
                "log1p_pearson_callable_transcripts",
                "mean_obs_mean",
                "median_obs_mean",
                "mean_pred_mean",
                "median_pred_mean",
                "mean_r",
                "median_r",
                "fisher_mean_r",
                "bp_weighted_mean_r",
                "mean_log1p_r",
                "median_log1p_r",
                "fisher_mean_log1p_r",
                "bp_weighted_mean_log1p_r",
            ],
        )

        filtered_codon_transcript_r_path = (
            f"{args.out_prefix}.gtf_filtered_per_transcript_codon_r.csv"
        )
        write_csv(
            filtered_codon_transcript_r_path,
            filtered_codon_transcript_r_rows,
            [
                "category",
                "expression_bin",
                "strand",
                "min_obs_mean_exclusive",
                "transcripts",
                "coverage_passing_transcripts",
                "pearson_callable_transcripts",
                "log1p_pearson_callable_transcripts",
                "mean_obs_mean",
                "median_obs_mean",
                "mean_pred_mean",
                "median_pred_mean",
                "mean_r",
                "median_r",
                "fisher_mean_r",
                "codon_weighted_mean_r",
                "mean_log1p_r",
                "median_log1p_r",
                "fisher_mean_log1p_r",
                "codon_weighted_mean_log1p_r",
            ],
        )

    plot_paths = []
    if not args.no_plots:
        try:
            write_plots(
                args.plots_dir,
                args.out_prefix,
                metric_rows,
                gtf_summary_rows,
                stop_profile_rows,
                gtf_phase_summary_rows,
                start_codon_frame_summary_rows,
                topk_rows,
                ranking_rows,
                cds_correlation_rows,
                filtered_region_transcript_r_rows,
                filtered_codon_transcript_r_rows,
                args.utr5_start_exclusion_bp,
                gtf_start_summary_rows=gtf_start_summary_rows,
                start_profile_rows=start_profile_rows,
            )
            prefix = Path(args.out_prefix).name
            for suffix in (
                "metrics",
                "calibration",
                "gtf_dropoff",
                "stop_profile",
                "gtf_start_boundary",
                "start_profile",
                "gtf_phase_agreement",
                "gtf_start_codon_frame_periodicity",
                "gtf_topk_overlap",
                "gtf_translation_scatter",
                "gtf_cds_position_correlations",
                "gtf_filtered_per_transcript_fisher_correlations",
                "gtf_utr5_start_exclusion_fisher_correlations",
            ):
                path = Path(args.plots_dir) / f"{prefix}.{suffix}.png"
                if path.exists():
                    plot_paths.append(str(path))
        except ImportError as exc:
            print(f"Skipping plots because matplotlib is not available: {exc}")

    summary_path = f"{args.out_prefix}.summary.txt"
    with open(summary_path, "w") as handle:
        handle.write("Ribo-seq prediction evaluation\n")
        handle.write(f"Regions: {args.regions_bed}\n")
        handle.write(
            "Chromosome filter: "
            f"{','.join(chroms) if chroms else 'none'}\n"
        )
        handle.write(f"Observed plus: {args.observed_plus}\n")
        handle.write(f"Observed minus: {args.observed_minus}\n")
        handle.write(f"Predicted plus: {args.predicted_plus}\n")
        handle.write(f"Predicted minus: {args.predicted_minus}\n")
        handle.write(f"Bin sizes: {','.join(map(str, bin_sizes))}\n")
        handle.write(f"Positive threshold: {args.positive_threshold}\n")
        handle.write(f"Rank metrics max points: {args.max_rank_points}\n")
        handle.write(
            "Filtered per-transcript r observed region mean cutoff: "
            f">{args.filtered_transcript_min_obs_mean}\n"
        )
        handle.write(
            "5'UTR start-proximal exclusion for utr5_distal metrics: "
            f"{args.utr5_start_exclusion_bp} bp\n"
        )
        handle.write(
            "5'UTR/CDS boundary thresholds and profile window: "
            f"min 5'UTR {args.min_utr5_bp} bp, "
            f"-{args.start_upstream}/+{args.start_downstream} bp\n"
        )
        handle.write(f"GTF: {args.gtf}\n")
        handle.write("\nMain files:\n")
        handle.write(f"- {metrics_path}\n")
        if gtf_summary_path:
            handle.write(f"- {gtf_summary_path}\n")
            handle.write(f"- {gtf_transcripts_path}\n")
            handle.write(f"- {stop_profile_path}\n")
        if gtf_start_summary_path:
            handle.write(f"- {gtf_start_summary_path}\n")
            handle.write(f"- {gtf_start_transcripts_path}\n")
            handle.write(f"- {start_profile_path}\n")
        if gtf_phase_path:
            handle.write(f"- {gtf_phase_path}\n")
            handle.write(f"- {gtf_phase_summary_path}\n")
            handle.write(f"- {start_codon_frame_path}\n")
            handle.write(f"- {start_codon_frame_summary_path}\n")
            handle.write(f"- {ranking_path}\n")
            handle.write(f"- {topk_path}\n")
            handle.write(f"- {cds_correlation_path}\n")
            handle.write(f"- {region_correlation_path}\n")
            handle.write(f"- {cds_codon_correlation_path}\n")
            handle.write(f"- {region_codon_correlation_path}\n")
            handle.write(f"- {filtered_transcript_r_path}\n")
            handle.write(f"- {filtered_region_transcript_r_path}\n")
            handle.write(f"- {filtered_codon_transcript_r_path}\n")
        if plot_paths:
            handle.write("\nPlots:\n")
            for plot_path in plot_paths:
                handle.write(f"- {plot_path}\n")

    print(f"Wrote {metrics_path}")
    if gtf_summary_path:
        print(f"Wrote {gtf_summary_path}")
        print(f"Wrote {gtf_transcripts_path}")
        print(f"Wrote {stop_profile_path}")
    if gtf_start_summary_path:
        print(f"Wrote {gtf_start_summary_path}")
        print(f"Wrote {gtf_start_transcripts_path}")
        print(f"Wrote {start_profile_path}")
    if gtf_phase_path:
        print(f"Wrote {gtf_phase_path}")
        print(f"Wrote {gtf_phase_summary_path}")
        print(f"Wrote {start_codon_frame_path}")
        print(f"Wrote {start_codon_frame_summary_path}")
        print(f"Wrote {ranking_path}")
        print(f"Wrote {topk_path}")
        print(f"Wrote {cds_correlation_path}")
        print(f"Wrote {region_correlation_path}")
        print(f"Wrote {cds_codon_correlation_path}")
        print(f"Wrote {region_codon_correlation_path}")
        print(f"Wrote {filtered_transcript_r_path}")
        print(f"Wrote {filtered_region_transcript_r_path}")
        print(f"Wrote {filtered_codon_transcript_r_path}")
    for plot_path in plot_paths:
        print(f"Wrote {plot_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
