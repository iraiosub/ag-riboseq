from pathlib import Path

import numpy as np
import pyBigWig


def parse_bigwig_paths(value):
    paths = [path.strip() for path in str(value).split(",") if path.strip()]
    if not paths:
        raise ValueError("Expected at least one BigWig path")
    return paths


class BigWigReplicates:
    def __init__(self, paths):
        self.paths = parse_bigwig_paths(paths)
        self.handles = []
        for path in self.paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Missing BigWig file: {path}")
            handle = pyBigWig.open(path)
            if handle is None:
                raise OSError(f"Could not open BigWig file: {path}")
            self.handles.append(handle)

    def values(self, chrom, start, end):
        total = None
        for handle in self.handles:
            values = np.asarray(handle.values(chrom, start, end), dtype=np.float32)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            total = values if total is None else total + values
        return total / len(self.handles)

    def stats(self, chrom, start, end, *args, **kwargs):
        values = []
        for handle in self.handles:
            value = handle.stats(chrom, start, end, *args, **kwargs)[0]
            values.append(0.0 if value is None else float(value))
        return [float(np.mean(values))]

    def close(self):
        for handle in self.handles:
            handle.close()
