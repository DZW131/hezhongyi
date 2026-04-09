from pathlib import Path
from typing import Any, Optional, Union

import torch


def load_torch_state(path: Union[str, Path], map_location: Optional[Union[str, torch.device]] = None) -> Any:
    """
    Load a torch checkpoint while preferring weights_only=True on newer PyTorch
    builds and remaining compatible with older versions.
    """
    load_kwargs = {}
    if map_location is not None:
        load_kwargs['map_location'] = map_location

    try:
        return torch.load(path, weights_only=True, **load_kwargs)
    except TypeError:
        return torch.load(path, **load_kwargs)
