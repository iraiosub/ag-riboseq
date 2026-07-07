import argparse
import os
from pathlib import Path

import numpy as np
import pyfaidx
import torch
import torch.nn as nn
from bigwig_replicates import BigWigReplicates
from riboseq_model import RIBOSEQ_HEAD_MODALITY, build_riboseq_model, parse_dilations
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SEQ_LEN = 1048576
PROJECT_DIR = "/nemo/project/proj-ai-dna-hackathon/proj5"
RIBO_BIGWIG_DIR = f"{PROJECT_DIR}/bigwig"
BW_PLUS = ",".join([
    f"{RIBO_BIGWIG_DIR}/human_brain_ribo_1.psites.forward.rpm.bigWig",
    f"{RIBO_BIGWIG_DIR}/human_brain_ribo_2.psites.forward.rpm.bigWig",
])
BW_MINUS = ",".join([
    f"{RIBO_BIGWIG_DIR}/human_brain_ribo_1.psites.reverse.rpm.bigWig",
    f"{RIBO_BIGWIG_DIR}/human_brain_ribo_2.psites.reverse.rpm.bigWig",
])
FASTA = f"{PROJECT_DIR}/ref/GRCh38.primary_assembly.genome.fa"
TRAIN_BED = f"{PROJECT_DIR}/regions/ag_fold0/train.bed"
VALID_BED = f"{PROJECT_DIR}/regions/ag_fold0/valid.bed"
CHECKPOINT_PATH = f"{PROJECT_DIR}/alphagenome_riboseq_head_ag_fold0.pth"
TRUNK_CHECKPOINT = f"{PROJECT_DIR}/models/model_fold_0.safetensors"
ALPHAGENOME_MODEL_VERSION = "fold_0"
LOG_RATE_MIN = -20.0
LOG_RATE_MAX = 8.0
MULTINOMIAL_SEGMENT_BP = 2048
POSITIONAL_WEIGHT = 5.0
COUNT_WEIGHT = 1.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune an AlphaGenome transfer head on Ribo-seq P-site tracks."
    )
    parser.add_argument(
        "--bw-plus",
        default=BW_PLUS,
        help="Plus-strand target BigWig path(s), comma-separated for replicate averaging.",
    )
    parser.add_argument(
        "--bw-minus",
        default=BW_MINUS,
        help="Minus-strand target BigWig path(s), comma-separated for replicate averaging.",
    )
    parser.add_argument("--fasta", default=FASTA)
    parser.add_argument("--train-bed", default=TRAIN_BED)
    parser.add_argument("--valid-bed", default=VALID_BED)
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--trunk-checkpoint", default=TRUNK_CHECKPOINT)
    parser.add_argument("--alphagenome-model-version", default=ALPHAGENOME_MODEL_VERSION)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--head-architecture", choices=["linear", "dilated"], default="linear")
    parser.add_argument("--dilated-hidden-channels", type=int, default=32)
    parser.add_argument("--dilated-kernel-size", type=int, default=9)
    parser.add_argument("--dilated-dilations", default="1,2,4,8")
    parser.add_argument("--train-max-windows", type=int, default=None)
    parser.add_argument("--valid-max-windows", type=int, default=None)
    parser.add_argument(
        "--region-selection",
        choices=["first", "signal-mixed"],
        default="first",
        help="How to downsample BED windows when --*-max-windows is set.",
    )
    parser.add_argument("--signal-high-fraction", type=float, default=0.75)
    parser.add_argument(
        "--loss-mode",
        choices=[
            "direct_poisson",
            "weighted_direct_poisson",
            "weighted_log_poisson",
            "poisson_multinomial",
        ],
        default="direct_poisson",
    )
    parser.add_argument(
        "--multinomial-segment-bp",
        type=int,
        default=MULTINOMIAL_SEGMENT_BP,
        help="Segment length for the Poisson-multinomial profile/count loss.",
    )
    parser.add_argument(
        "--positional-weight",
        type=float,
        default=POSITIONAL_WEIGHT,
        help="Weight for the multinomial positional/profile loss term.",
    )
    parser.add_argument(
        "--count-weight",
        type=float,
        default=COUNT_WEIGHT,
        help="Weight for the Poisson segment-total loss term.",
    )
    parser.add_argument(
        "--nonzero-weight",
        type=float,
        default=4.0,
        help="Extra per-base weight added where observed P-site signal is >0.",
    )
    parser.add_argument(
        "--signal-weight",
        type=float,
        default=1.0,
        help="Extra per-base weight proportional to log1p(observed P-site signal).",
    )
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    return parser.parse_args()


def load_bed_regions(path, seq_len=SEQ_LEN):
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
            start = int(start)
            end = int(end)
            if end - start != seq_len:
                raise ValueError(
                    f"Region {(chrom, start, end)} has length {end - start}; "
                    f"expected {seq_len}"
                )
            regions.append((chrom, start, end))
    if not regions:
        raise ValueError(f"No regions found in {path}")
    return regions


def ensure_file(path, description):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def ensure_region_file(path):
    ensure_file(path, "region file")


def ensure_fasta_index(path):
    ensure_file(path, "FASTA")
    ensure_file(f"{path}.fai", "FASTA index")


def load_checkpoint_state(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def bw_sum(bigwig, chrom, start, end):
    value = bigwig.stats(chrom, start, end, type="sum", exact=True)[0]
    return 0.0 if value is None else float(value)


def score_regions_by_signal(regions, bigwig_plus, bigwig_minus):
    scored_regions = []
    plus = BigWigReplicates(bigwig_plus)
    minus = BigWigReplicates(bigwig_minus)
    try:
        for region in tqdm(regions, desc="Scoring windows"):
            chrom, start, end = region
            signal = bw_sum(plus, chrom, start, end) + bw_sum(minus, chrom, start, end)
            scored_regions.append((signal, region))
    finally:
        plus.close()
        minus.close()
    return scored_regions


def evenly_sample(items, count):
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    indexes = np.linspace(0, len(items) - 1, num=count, dtype=int)
    sampled = []
    seen = set()
    for index in indexes:
        if index not in seen:
            sampled.append(items[index])
            seen.add(index)
    for index, item in enumerate(items):
        if len(sampled) >= count:
            break
        if index not in seen:
            sampled.append(item)
    return sampled


def select_regions(regions, max_windows, selection, bigwig_plus, bigwig_minus, high_fraction):
    if max_windows is None or max_windows >= len(regions):
        return regions
    if max_windows <= 0:
        raise ValueError("--*-max-windows must be positive when provided")

    if selection == "first":
        return regions[:max_windows]

    scored = score_regions_by_signal(regions, bigwig_plus, bigwig_minus)
    scored.sort(key=lambda item: item[0], reverse=True)

    high_count = int(round(max_windows * high_fraction))
    high_count = min(max(high_count, 1), max_windows)
    high_regions = [region for _signal, region in scored[:high_count]]
    high_region_set = set(high_regions)
    remaining_regions = [
        region for _signal, region in sorted(scored[high_count:], key=lambda item: item[1])
        if region not in high_region_set
    ]
    background_regions = evenly_sample(remaining_regions, max_windows - len(high_regions))
    selected = high_regions + background_regions
    return selected[:max_windows]


def forward_riboseq(model, dna_onehot, organism_index, device):
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        preds = model(
            dna_onehot,
            organism_index=organism_index,
            heads=["ribo_seq"],
            resolutions=(1,),
            return_scaled_predictions=False,
        )
    return preds["ribo_seq"][1]


class RiboSeqLoss(nn.Module):
    def __init__(
        self,
        mode,
        nonzero_weight,
        signal_weight,
        max_weight,
        multinomial_segment_bp=MULTINOMIAL_SEGMENT_BP,
        positional_weight=POSITIONAL_WEIGHT,
        count_weight=COUNT_WEIGHT,
        log_rate_min=LOG_RATE_MIN,
        log_rate_max=LOG_RATE_MAX,
    ):
        super().__init__()
        self.mode = mode
        self.nonzero_weight = nonzero_weight
        self.signal_weight = signal_weight
        self.max_weight = max_weight
        self.multinomial_segment_bp = multinomial_segment_bp
        self.positional_weight = positional_weight
        self.count_weight = count_weight
        self.log_rate_min = log_rate_min
        self.log_rate_max = log_rate_max
        self.direct_poisson = nn.PoissonNLLLoss(log_input=False, reduction="none")
        self.log_poisson = nn.PoissonNLLLoss(log_input=True, reduction="none")

    @property
    def output_is_log_rate(self):
        return self.mode == "weighted_log_poisson"

    def prediction_rate(self, prediction):
        prediction = prediction.float()
        if self.output_is_log_rate:
            return torch.exp(torch.clamp(prediction, self.log_rate_min, self.log_rate_max))
        return torch.clamp(prediction, min=1e-8)

    def target_weights(self, target):
        weights = torch.ones_like(target)
        if self.nonzero_weight:
            weights = weights + (target > 0).to(target.dtype) * self.nonzero_weight
        if self.signal_weight:
            weights = weights + torch.log1p(target) * self.signal_weight
        return torch.clamp(weights, min=1.0, max=self.max_weight)

    def to_channels_last(self, values, name):
        if values.ndim != 3:
            raise ValueError(f"Expected 3D {name} tensor, got {values.shape}")
        if values.shape[-1] == 2:
            return values.float()
        if values.shape[1] == 2:
            return values.transpose(1, 2).contiguous().float()
        raise ValueError(f"Expected two-track {name} tensor, got {values.shape}")

    def safe_mean(self, values, mask=None):
        if mask is None:
            return values.mean()
        mask = mask.expand_as(values).float()
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def poisson_zero_min_loss(self, y_true, y_pred, mask):
        y_true = torch.clamp(y_true.float(), min=0.0)
        y_pred = torch.clamp(y_pred.float(), min=1e-7)
        min_value = y_true - y_true * torch.log(y_true + 1e-7)
        loss = (y_pred - y_true * torch.log(y_pred)) - min_value
        return self.safe_mean(loss, mask)

    def poisson_multinomial_loss(self, prediction, target):
        prediction = self.to_channels_last(self.prediction_rate(prediction), "prediction")
        target = torch.clamp(self.to_channels_last(target, "target"), min=0.0)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes must match, got {prediction.shape} and {target.shape}"
            )

        segment_bp = self.multinomial_segment_bp
        if segment_bp <= 0:
            raise ValueError("--multinomial-segment-bp must be positive")
        seq_len = prediction.shape[-2]
        if seq_len % segment_bp != 0:
            raise ValueError(
                f"Prediction length {seq_len} must be divisible by "
                f"--multinomial-segment-bp {segment_bp}"
            )

        batch_dims = target.shape[:-2]
        channels = target.shape[-1]
        num_segments = seq_len // segment_bp
        mask = torch.ones(*batch_dims, 1, channels, dtype=torch.bool, device=target.device)

        target = target.reshape(*batch_dims, num_segments, segment_bp, channels)
        prediction = prediction.reshape(*batch_dims, num_segments, segment_bp, channels)

        target_total = target.sum(dim=-2, keepdim=True, dtype=torch.float32)
        pred_total = prediction.sum(dim=-2, keepdim=True, dtype=torch.float32)
        mask_expanded = mask.unsqueeze(-2)

        count_loss = self.poisson_zero_min_loss(
            y_true=target_total,
            y_pred=pred_total,
            mask=mask_expanded,
        )
        count_loss = count_loss / segment_bp

        probabilities = prediction.float() / (pred_total + 1e-7)
        positional_loss = -target * torch.log(probabilities + 1e-7)
        positional_loss = self.safe_mean(positional_loss, mask=mask_expanded)

        return self.count_weight * count_loss + self.positional_weight * positional_loss

    def forward(self, prediction, target):
        target = target.float()
        if self.mode == "poisson_multinomial":
            return self.poisson_multinomial_loss(prediction, target)

        if self.output_is_log_rate:
            loss_input = torch.clamp(
                prediction.float(),
                self.log_rate_min,
                self.log_rate_max,
            )
            per_base_loss = self.log_poisson(loss_input, target)
        else:
            loss_input = torch.clamp(prediction.float(), min=1e-8)
            per_base_loss = self.direct_poisson(loss_input, target)

        if self.mode == "direct_poisson":
            return per_base_loss.mean()

        weights = self.target_weights(target)
        return (per_base_loss * weights).sum() / weights.sum().clamp_min(1.0)


def evaluate(model, dataloader, loss_fn, device, desc):
    model.eval()
    total_loss = 0
    total_pred_mean = 0
    total_target_mean = 0

    with torch.inference_mode():
        pbar = tqdm(dataloader, desc=desc)
        for dna_onehot, ribo_targets in pbar:
            dna_onehot = dna_onehot.to(device)
            ribo_targets = ribo_targets.to(device)
            organism_index = torch.zeros(
                dna_onehot.shape[0],
                dtype=torch.long,
                device=device,
            )

            ribo_preds = forward_riboseq(model, dna_onehot, organism_index, device)
            loss = loss_fn(ribo_preds, ribo_targets)
            pred_mean = loss_fn.prediction_rate(ribo_preds).mean().item()
            target_mean = ribo_targets.float().mean().item()

            total_loss += loss.item()
            total_pred_mean += pred_mean
            total_target_mean += target_mean
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "PredMean": f"{pred_mean:.4f}",
                "TargetMean": f"{target_mean:.4f}",
            })

    num_batches = len(dataloader)
    return (
        total_loss / num_batches,
        total_pred_mean / num_batches,
        total_target_mean / num_batches,
    )


class RiboSeqDataset(Dataset):
    def __init__(self, bigwig_plus, bigwig_minus, fasta_file, bed_regions, seq_len=SEQ_LEN):
        self.bw_plus = BigWigReplicates(bigwig_plus)
        self.bw_minus = BigWigReplicates(bigwig_minus)
        self.fasta = pyfaidx.Fasta(fasta_file)
        self.regions = bed_regions
        self.seq_len = seq_len

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        chrom, start, end = self.regions[idx]
        if end - start != self.seq_len:
            raise ValueError(
                f"Region {(chrom, start, end)} has length {end - start}; "
                f"expected {self.seq_len}"
            )

        seq = self.fasta[chrom][start:end].seq.upper()
        seq_array = np.array(list(seq))
        onehot = np.zeros((self.seq_len, 4), dtype=np.float32)
        onehot[seq_array == "A"] = [1, 0, 0, 0]
        onehot[seq_array == "C"] = [0, 1, 0, 0]
        onehot[seq_array == "G"] = [0, 0, 1, 0]
        onehot[seq_array == "T"] = [0, 0, 0, 1]

        plus_vals = np.nan_to_num(self.bw_plus.values(chrom, start, end))
        minus_vals = np.nan_to_num(self.bw_minus.values(chrom, start, end))
        ribo_targets = np.column_stack((plus_vals, minus_vals))

        dna_tensor = torch.tensor(onehot, dtype=torch.float32)
        ribo_tensor = torch.tensor(ribo_targets, dtype=torch.float32)
        return dna_tensor, ribo_tensor


def main():
    args = parse_args()
    ensure_file(args.trunk_checkpoint, "AlphaGenome trunk checkpoint")
    ensure_fasta_index(args.fasta)
    ensure_region_file(args.train_bed)
    ensure_region_file(args.valid_bed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type != "cuda" and os.environ.get("ALLOW_CPU", "0") != "1":
        raise RuntimeError(
            "CUDA is not available, so training would run on CPU. "
            "Submit with a GPU allocation and use a CUDA-enabled PyTorch env, "
            "or set ALLOW_CPU=1 if you really want a CPU debug run."
        )

    print("Loading AlphaGenome trunk...", flush=True)
    model = build_riboseq_model(
        load_trunk_path=args.trunk_checkpoint,
        head_architecture=args.head_architecture,
        head_modality=RIBOSEQ_HEAD_MODALITY,
        dilated_hidden_channels=args.dilated_hidden_channels,
        dilated_kernel_size=args.dilated_kernel_size,
        dilated_dilations=parse_dilations(args.dilated_dilations),
    )
    if args.resume_from_checkpoint:
        print(f"Loading checkpoint weights from {args.resume_from_checkpoint}", flush=True)
        model.load_state_dict(load_checkpoint_state(args.resume_from_checkpoint), strict=True)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )
    loss_fn = RiboSeqLoss(
        mode=args.loss_mode,
        nonzero_weight=args.nonzero_weight,
        signal_weight=args.signal_weight,
        max_weight=args.max_weight,
        multinomial_segment_bp=args.multinomial_segment_bp,
        positional_weight=args.positional_weight,
        count_weight=args.count_weight,
    )

    train_regions = load_bed_regions(args.train_bed)
    valid_regions = load_bed_regions(args.valid_bed)
    train_regions = select_regions(
        train_regions,
        args.train_max_windows,
        args.region_selection,
        args.bw_plus,
        args.bw_minus,
        args.signal_high_fraction,
    )
    valid_regions = select_regions(
        valid_regions,
        args.valid_max_windows,
        args.region_selection,
        args.bw_plus,
        args.bw_minus,
        args.signal_high_fraction,
    )
    print(f"Training windows: {len(train_regions)} from {args.train_bed}", flush=True)
    print(f"Validation windows: {len(valid_regions)} from {args.valid_bed}", flush=True)
    print(f"AlphaGenome trunk checkpoint: {args.trunk_checkpoint}", flush=True)
    print(f"AlphaGenome model version: {args.alphagenome_model_version}", flush=True)
    print(f"Ribo-seq head modality: {RIBOSEQ_HEAD_MODALITY}", flush=True)
    print(f"Learning rate: {args.learning_rate}", flush=True)
    print(f"Head architecture: {args.head_architecture}", flush=True)
    if args.head_architecture == "dilated":
        receptive_field = 1 + (args.dilated_kernel_size - 1) * sum(
            parse_dilations(args.dilated_dilations)
        )
        print(
            "Dilated head: "
            f"hidden={args.dilated_hidden_channels}, "
            f"kernel={args.dilated_kernel_size}, "
            f"dilations={args.dilated_dilations}, "
            f"receptive_field={receptive_field} bp",
            flush=True,
        )
    print(f"Loss mode: {args.loss_mode}", flush=True)
    if args.loss_mode == "poisson_multinomial":
        print(
            "Poisson-multinomial loss: "
            f"segment_bp={args.multinomial_segment_bp}, "
            f"positional_weight={args.positional_weight}, "
            f"count_weight={args.count_weight}",
            flush=True,
        )
        print(
            "Poisson-multinomial predictions are interpreted as rates, not log-rates.",
            flush=True,
        )
    elif args.loss_mode != "direct_poisson":
        print(
            "Target weighting: "
            f"nonzero +{args.nonzero_weight}, "
            f"log1p signal x{args.signal_weight}, "
            f"max {args.max_weight}",
            flush=True,
        )

    train_dataset = RiboSeqDataset(args.bw_plus, args.bw_minus, args.fasta, train_regions)
    valid_dataset = RiboSeqDataset(args.bw_plus, args.bw_minus, args.fasta, valid_regions)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    accumulation_steps = args.accumulation_steps
    epochs = args.epochs
    early_stopping_patience = args.early_stopping_patience
    min_delta = args.min_delta
    checkpoint_path = args.checkpoint
    best_valid_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print("Starting training...", flush=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_pred_mean = 0
        epoch_target_mean = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_dataloader, desc=f"Train {epoch + 1}/{epochs}")
        num_batches = len(train_dataloader)
        for i, (dna_onehot, ribo_targets) in enumerate(pbar):
            dna_onehot = dna_onehot.to(device)
            ribo_targets = ribo_targets.to(device)
            organism_index = torch.zeros(
                dna_onehot.shape[0],
                dtype=torch.long,
                device=device,
            )

            ribo_preds = forward_riboseq(model, dna_onehot, organism_index, device)
            raw_loss = loss_fn(ribo_preds, ribo_targets)

            group_start = (i // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, num_batches - group_start)
            loss = raw_loss / group_size
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == num_batches:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            batch_pred_mean = loss_fn.prediction_rate(ribo_preds.detach()).mean().item()
            batch_target_mean = ribo_targets.float().mean().item()
            epoch_loss += raw_loss.item()
            epoch_pred_mean += batch_pred_mean
            epoch_target_mean += batch_target_mean
            pbar.set_postfix({
                "Loss": f"{raw_loss.item():.4f}",
                "PredMean": f"{batch_pred_mean:.4f}",
                "TargetMean": f"{batch_target_mean:.4f}",
            })

        average_loss = epoch_loss / len(train_dataloader)
        average_pred_mean = epoch_pred_mean / len(train_dataloader)
        average_target_mean = epoch_target_mean / len(train_dataloader)
        print(
            f"Epoch {epoch + 1} Train Loss: {average_loss:.6f} "
            f"PredMean: {average_pred_mean:.6f} "
            f"TargetMean: {average_target_mean:.6f}",
            flush=True,
        )

        valid_loss, valid_pred_mean, valid_target_mean = evaluate(
            model,
            valid_dataloader,
            loss_fn,
            device,
            desc=f"Valid {epoch + 1}/{epochs}",
        )
        print(
            f"Epoch {epoch + 1} Valid Loss: {valid_loss:.6f} "
            f"PredMean: {valid_pred_mean:.6f} "
            f"TargetMean: {valid_target_mean:.6f}",
            flush=True,
        )

        if valid_loss < best_valid_loss - min_delta:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "metadata": {
                        "loss_mode": args.loss_mode,
                        "prediction_is_log_rate": loss_fn.output_is_log_rate,
                        "head_architecture": args.head_architecture,
                        "dilated_hidden_channels": args.dilated_hidden_channels,
                        "dilated_kernel_size": args.dilated_kernel_size,
                        "dilated_dilations": args.dilated_dilations,
                        "seq_len": SEQ_LEN,
                        "modality": RIBOSEQ_HEAD_MODALITY,
                        "num_tracks": 2,
                        "resolutions": [1],
                        "trunk_checkpoint": args.trunk_checkpoint,
                        "alphagenome_model_version": args.alphagenome_model_version,
                        "bw_plus": args.bw_plus,
                        "bw_minus": args.bw_minus,
                        "train_bed": args.train_bed,
                        "valid_bed": args.valid_bed,
                        "train_windows": len(train_regions),
                        "valid_windows": len(valid_regions),
                        "best_epoch": best_epoch,
                        "best_valid_loss": best_valid_loss,
                        "learning_rate": args.learning_rate,
                        "resume_from_checkpoint": args.resume_from_checkpoint,
                        "nonzero_weight": args.nonzero_weight,
                        "signal_weight": args.signal_weight,
                        "max_weight": args.max_weight,
                        "multinomial_segment_bp": args.multinomial_segment_bp,
                        "positional_weight": args.positional_weight,
                        "count_weight": args.count_weight,
                    },
                },
                checkpoint_path,
            )
            print(f"Saved new best checkpoint to {checkpoint_path}", flush=True)
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement}/"
                f"{early_stopping_patience} epochs",
                flush=True,
            )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Stopping early at epoch {epoch + 1}. "
                f"Best epoch was {best_epoch} with validation loss {best_valid_loss:.6f}.",
                flush=True,
            )
            break

    print(
        f"Training complete. Best weights saved to {checkpoint_path} "
        f"from epoch {best_epoch} with validation loss {best_valid_loss:.6f}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
