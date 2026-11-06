#!/usr/bin/env python3
"""Entry point for Chemprop 1.6 training.

Chemprop 1.6 serialises its own argparse.Namespace inside model.pt. PyTorch
2.6 changed torch.load's default to weights_only=True, which rejects that
checkpoint before Chemprop can reload it. Restoring the previous default here
only affects checkpoints this project writes itself.
"""
import torch

_torch_load = torch.load


def _chemprop_compatible_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _chemprop_compatible_load

from chemprop.train import chemprop_train  # noqa: E402

if __name__ == "__main__":
    chemprop_train()
