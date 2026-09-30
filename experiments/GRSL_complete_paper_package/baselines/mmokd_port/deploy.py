from __future__ import annotations

from pathlib import Path

import torch

from models.landslide_cocd import ConvNeXtTinyFPN
from mmokd_model import SingleOrbitBranch, load_haiti_initialization


def load_student(branch: str, checkpoint: str | Path, initialization: str | Path,
                 device: str | torch.device = "cpu") -> SingleOrbitBranch:
    """Load one deployable orbit branch; it constructs no paired Teacher."""
    if branch not in ("asc", "desc"):
        raise ValueError("branch must be 'asc' or 'desc'")
    template = ConvNeXtTinyFPN()
    load_haiti_initialization(template, Path(initialization))
    model = SingleOrbitBranch(template)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
