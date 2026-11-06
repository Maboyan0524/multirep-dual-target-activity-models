#!/usr/bin/env python3
"""Entry point for Chemprop 1.6 prediction.

See chemprop_train_entry.py: this restores Chemprop 1.6's ability to load the
checkpoints written by this project under PyTorch 2.6 and later.
"""
import torch

_torch_load = torch.load


def _chemprop_compatible_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _chemprop_compatible_load

from chemprop.train import chemprop_predict  # noqa: E402

if __name__ == "__main__":
    chemprop_predict()
