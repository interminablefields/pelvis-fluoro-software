from typing import TypeVar, Union, Optional, Any, Callable, Tuple, NamedTuple
import logging
import torch
import numpy as np

log = logging.getLogger(__name__)

T = TypeVar("T", torch.Tensor, np.ndarray)


def batchify(x: T, num_channels: int = 3) -> T:
    """Convert a single image to a batch of size 1.

    Args:
        x: Either an array of images or a single image. (H, W), (H,W,C) or (N,H,W,C)

    Returns:
        The batchified image. (N,H,W,C)

    """
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            x = np.expand_dims(x, axis=-1)
            x = np.repeat(x, num_channels, axis=-1)
        elif x.ndim == 3:
            x = np.expand_dims(x, axis=0)
        elif x.ndim == 4:
            return x
        else:
            raise ValueError(f"Invalid number of dimensions: {x.ndim}")
    elif isinstance(x, torch.Tensor):
        if x.ndim == 2:
            x = x.unsqueeze(dim=0)
            x = x.repeat(1, 1, 1, num_channels)
        elif x.ndim == 3:
            x = x.unsqueeze(dim=0)
        elif x.ndim == 4:
            return x
        else:
            raise ValueError(f"Invalid number of dimensions: {x.ndim}")
    else:
        raise TypeError(f"Invalid type: {type(x)}")

    if x.shape[-1] != num_channels:
        raise ValueError(f"Invalid number of channels: {x.shape[-1]}")

    return batchify(x)
