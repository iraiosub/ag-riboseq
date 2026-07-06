import math

import torch
import torch.nn as nn
from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.extensions.finetuning.transfer import (
    TransferConfig,
    load_trunk,
    prepare_for_transfer,
    remove_all_heads,
)


RIBOSEQ_HEAD_MODALITY = "rna_seq"


def parse_dilations(value):
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(value)


def _prediction_to_ncl(prediction):
    if prediction.ndim != 3:
        raise ValueError(f"Expected 3D prediction tensor, got {prediction.shape}")
    if prediction.shape[-1] == 2:
        return prediction.transpose(1, 2), "nlc"
    if prediction.shape[1] == 2:
        return prediction, "ncl"
    raise ValueError(f"Expected two-track prediction tensor, got {prediction.shape}")


def _prediction_from_ncl(prediction, layout):
    if layout == "nlc":
        return prediction.transpose(1, 2)
    if layout == "ncl":
        return prediction
    raise ValueError(f"Unknown prediction layout: {layout}")


def _dna_to_ncl(dna_sequence):
    if dna_sequence.ndim != 3:
        raise ValueError(f"Expected 3D DNA tensor, got {dna_sequence.shape}")
    if dna_sequence.shape[-1] == 4:
        return dna_sequence.transpose(1, 2)
    if dna_sequence.shape[1] == 4:
        return dna_sequence
    raise ValueError(f"Expected one-hot DNA tensor, got {dna_sequence.shape}")


def _center_crop(values, length):
    if values.shape[-1] == length:
        return values
    if values.shape[-1] < length:
        raise ValueError(f"Cannot crop length {values.shape[-1]} to longer length {length}")
    start = (values.shape[-1] - length) // 2
    return values[..., start:start + length]


class DilatedRiboSeqRefiner(nn.Module):
    def __init__(
        self,
        hidden_channels=32,
        kernel_size=9,
        dilations=(1, 2, 4, 8),
        log_rate_min=-20.0,
        log_rate_max=8.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("DilatedRiboSeqRefiner requires an odd kernel_size")

        self.log_rate_min = log_rate_min
        self.log_rate_max = log_rate_max
        self.max_rate = math.exp(log_rate_max)
        in_channels = 6  # log plus/minus prediction + A/C/G/T sequence.
        layers = []
        for dilation in parse_dilations(dilations):
            padding = (kernel_size // 2) * dilation
            layers.extend([
                nn.Conv1d(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.GELU(),
            ])
            in_channels = hidden_channels

        self.context = nn.Sequential(*layers)
        self.delta = nn.Conv1d(hidden_channels, 2, kernel_size=1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, base_prediction, dna_sequence):
        base_ncl, layout = _prediction_to_ncl(base_prediction)
        dna_ncl = _center_crop(_dna_to_ncl(dna_sequence), base_ncl.shape[-1])

        base_log_rate = torch.log(
            torch.clamp(base_ncl.float(), min=1e-8, max=self.max_rate)
        )
        features = torch.cat([base_log_rate, dna_ncl.to(base_log_rate.dtype)], dim=1)
        delta = self.delta(self.context(features))
        refined_log_rate = torch.clamp(
            base_log_rate + delta,
            min=self.log_rate_min,
            max=self.log_rate_max,
        )
        refined = torch.exp(refined_log_rate).to(base_prediction.dtype)
        return _prediction_from_ncl(refined, layout)


class DilatedRiboSeqModel(nn.Module):
    def __init__(self, base_model, refiner):
        super().__init__()
        self.base_model = base_model
        self.refiner = refiner

    def forward(self, dna_sequence, *args, **kwargs):
        predictions = self.base_model(dna_sequence, *args, **kwargs)
        ribo_predictions = dict(predictions["ribo_seq"])
        ribo_predictions[1] = self.refiner(ribo_predictions[1], dna_sequence)

        predictions = dict(predictions)
        predictions["ribo_seq"] = ribo_predictions
        return predictions


def build_riboseq_model(
    load_trunk_path=None,
    head_architecture="linear",
    head_modality=RIBOSEQ_HEAD_MODALITY,
    dilated_hidden_channels=32,
    dilated_kernel_size=9,
    dilated_dilations=(1, 2, 4, 8),
):
    model = AlphaGenome()
    if load_trunk_path is not None:
        model = load_trunk(model, load_trunk_path)
    model = remove_all_heads(model)

    config = TransferConfig(
        mode="linear",
        new_heads={
            "ribo_seq": {
                "modality": head_modality,
                "num_tracks": 2,
                "resolutions": [1],
            }
        },
    )
    model = prepare_for_transfer(model, config)

    if head_architecture == "linear":
        return model
    if head_architecture == "dilated":
        refiner = DilatedRiboSeqRefiner(
            hidden_channels=dilated_hidden_channels,
            kernel_size=dilated_kernel_size,
            dilations=parse_dilations(dilated_dilations),
        )
        return DilatedRiboSeqModel(model, refiner)
    raise ValueError(f"Unknown head architecture: {head_architecture}")
