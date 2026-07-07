import argparse
import os

import numpy as np
import pyBigWig
import pyfaidx
import torch
from riboseq_model import RIBOSEQ_HEAD_MODALITY, build_riboseq_model, parse_dilations
from tqdm import tqdm


SEQ_LEN = 1048576
DEFAULT_FASTA = "/camp/lab/ulej/home/shared/oscar_ira_riboloco/ref/human/GRCh38.primary_assembly.genome.fa"
DEFAULT_REGIONS = [
    ("chr2", 163727319, 164775895),
    ("chr2", 164775895, 165824471),
    ("chr2", 165824471, 166873047),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export trained AlphaGenome Ribo-seq predictions as IGV-loadable BigWigs."
    )
    parser.add_argument("--checkpoint", default="alphagenome_riboseq_head.pth")
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--regions-bed", default=None, help="Optional BED file with windows to predict.")
    parser.add_argument("--output-prefix", default="riboseq_pred")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--max-value", type=float, default=None)
    parser.add_argument("--min-value", type=float, default=0.0)
    parser.add_argument(
        "--input-is-log-rate",
        action="store_true",
        help="Treat model outputs as log-rates from an old checkpoint and exponentiate them.",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def load_regions(path):
    if path is None:
        return list(DEFAULT_REGIONS)

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
    return regions


def chrom_sizes_from_fasta(fasta):
    return [(chrom, len(fasta[chrom])) for chrom in fasta.keys()]


def sort_and_validate_regions(regions, chrom_sizes):
    chrom_rank = {chrom: i for i, (chrom, _length) in enumerate(chrom_sizes)}
    missing = sorted({chrom for chrom, _start, _end in regions if chrom not in chrom_rank})
    if missing:
        raise ValueError(f"Regions reference chromosomes missing from FASTA: {missing[:5]}")

    regions = sorted(regions, key=lambda region: (chrom_rank[region[0]], region[1], region[2]))
    last_end = {}
    for chrom, start, end in regions:
        if start < 0 or end <= start:
            raise ValueError(f"Invalid region: {(chrom, start, end)}")
        if start < last_end.get(chrom, 0):
            raise ValueError(
                "BigWig output requires non-overlapping sorted regions. "
                f"Found overlap around {(chrom, start, end)}."
            )
        last_end[chrom] = end
    return regions


def one_hot_sequence(fasta, chrom, start, end, seq_len):
    seq = fasta[chrom][start:end].seq.upper()
    if len(seq) != seq_len:
        raise ValueError(
            f"Region {(chrom, start, end)} produced {len(seq)} bp, expected {seq_len} bp."
        )

    seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    onehot = np.zeros((seq_len, 4), dtype=np.float32)
    onehot[seq_bytes == ord("A"), 0] = 1.0
    onehot[seq_bytes == ord("C"), 1] = 1.0
    onehot[seq_bytes == ord("G"), 2] = 1.0
    onehot[seq_bytes == ord("T"), 3] = 1.0
    return torch.from_numpy(onehot).unsqueeze(0)


def load_checkpoint(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    metadata = {}
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        metadata = checkpoint.get("metadata", {})
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    return state_dict, metadata


def build_model(checkpoint_path, device):
    state_dict, metadata = load_checkpoint(checkpoint_path)
    head_modality = metadata.get("modality", RIBOSEQ_HEAD_MODALITY)
    model = build_riboseq_model(
        head_architecture=metadata.get("head_architecture", "linear"),
        head_modality=head_modality,
        dilated_hidden_channels=metadata.get("dilated_hidden_channels", 32),
        dilated_kernel_size=metadata.get("dilated_kernel_size", 9),
        dilated_dilations=parse_dilations(metadata.get("dilated_dilations", "1,2,4,8")),
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print(f"Ribo-seq head modality: {head_modality}")
    return model, metadata


def split_prediction(prediction):
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.ndim != 2:
        raise ValueError(f"Expected 2D prediction after batch squeeze, got shape {prediction.shape}")
    if prediction.shape[-1] == 2:
        return prediction[:, 0], prediction[:, 1]
    if prediction.shape[0] == 2:
        return prediction[0, :], prediction[1, :]
    raise ValueError(f"Expected two Ribo-seq tracks, got prediction shape {prediction.shape}")


def to_track_values(values, max_value, min_value, input_is_log_rate):
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if input_is_log_rate:
        values = np.exp(np.clip(values, -20.0, 8.0))
    values = np.clip(values, 0.0, None)
    if min_value > 0:
        values = np.where(values >= min_value, values, 0.0)
    if max_value is not None:
        values = np.clip(values, 0.0, max_value)
    return values.astype(np.float32)


def write_fixed_step(bigwig, chrom, start, values, chunk_size=100000):
    for offset in range(0, len(values), chunk_size):
        chunk = values[offset : offset + chunk_size]
        bigwig.addEntries(
            chrom,
            start + offset,
            values=chunk.tolist(),
            span=1,
            step=1,
        )


def prediction_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return "empty"
    nonzero = int(np.count_nonzero(values))
    return (
        f"min={np.min(values):.6g} "
        f"mean={np.mean(values):.6g} "
        f"max={np.max(values):.6g} "
        f"nonzero={nonzero}/{len(values)}"
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda" and not args.allow_cpu and os.environ.get("ALLOW_CPU", "0") != "1":
        raise RuntimeError("CUDA is not available. Use a GPU job or pass --allow-cpu for debugging.")

    fasta = pyfaidx.Fasta(args.fasta)
    chrom_sizes = chrom_sizes_from_fasta(fasta)
    regions = sort_and_validate_regions(load_regions(args.regions_bed), chrom_sizes)
    model, metadata = build_model(args.checkpoint, device)
    input_is_log_rate = (
        args.input_is_log_rate
        or bool(metadata.get("prediction_is_log_rate"))
    )
    if input_is_log_rate:
        print("Treating model outputs as log-rates and exponentiating before BigWig export.")

    plus_path = f"{args.output_prefix}.plus.bigwig"
    minus_path = f"{args.output_prefix}.minus.bigwig"
    plus_bw = pyBigWig.open(plus_path, "w")
    minus_bw = pyBigWig.open(minus_path, "w")
    plus_bw.addHeader(chrom_sizes)
    minus_bw.addHeader(chrom_sizes)
    raw_stats_printed = False

    try:
        for chrom, start, end in tqdm(regions, desc="Predicting"):
            dna_onehot = one_hot_sequence(fasta, chrom, start, end, args.seq_len).to(device)
            organism_index = torch.zeros(dna_onehot.shape[0], dtype=torch.long, device=device)

            with torch.inference_mode():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    preds = model(
                        dna_onehot,
                        organism_index=organism_index,
                        heads=["ribo_seq"],
                        resolutions=(1,),
                        return_scaled_predictions=False,
                    )
                    prediction = preds["ribo_seq"][1].float().cpu().numpy()[0]

            plus_pred, minus_pred = split_prediction(prediction)
            pred_len = len(plus_pred)
            if pred_len > (end - start):
                raise ValueError(
                    f"Prediction length {pred_len} is longer than region length {end - start}."
                )
            write_start = start + ((end - start) - pred_len) // 2

            plus_values = to_track_values(
                plus_pred,
                args.max_value,
                args.min_value,
                input_is_log_rate,
            )
            minus_values = to_track_values(
                minus_pred,
                args.max_value,
                args.min_value,
                input_is_log_rate,
            )
            if not raw_stats_printed:
                print(f"Raw plus prediction stats: {prediction_stats(plus_pred)}")
                print(f"Raw minus prediction stats: {prediction_stats(minus_pred)}")
                print(f"Export plus value stats: {prediction_stats(plus_values)}")
                print(f"Export minus value stats: {prediction_stats(minus_values)}")
                raw_stats_printed = True
            write_fixed_step(plus_bw, chrom, write_start, plus_values)
            write_fixed_step(minus_bw, chrom, write_start, minus_values)
    finally:
        plus_bw.close()
        minus_bw.close()
        fasta.close()

    print(f"Wrote {plus_path}")
    print(f"Wrote {minus_path}")


if __name__ == "__main__":
    main()
