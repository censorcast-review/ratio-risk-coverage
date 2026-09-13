"""Separate recomputed outputs from immutable publication evidence."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def safe_output(path):
    """Allow a new output below reproduction_outputs or outside the source tree."""
    p=Path(path).resolve()
    base=ROOT/'reproduction_outputs'
    allowed=base.resolve()
    if allowed!=base:
        raise ValueError('reproduction_outputs must not redirect through a symlink into archived evidence.')
    if p==ROOT or p in ROOT.parents or (ROOT in p.parents and p!=allowed and allowed not in p.parents):
        raise ValueError('Output must be under reproduction_outputs or outside the source tree; archived evidence is protected.')
    return p
