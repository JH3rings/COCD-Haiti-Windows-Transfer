"""Portable final dataset paths and device selection."""
import os
from pathlib import Path
import torch
PACKAGE=Path(__file__).resolve().parent.parent
DATASET_ROOT=Path(os.environ.get('COCD_DATASET_ROOT',PACKAGE/'data/haiti')).resolve()
SPLIT_DIR=Path(os.environ.get('COCD_SPLIT_DIR',PACKAGE/'data/splits_v3')).resolve()
OUT_ROOT=Path(os.environ.get('COCD_OUT_ROOT',PACKAGE/'artifacts')).resolve()
PROCESSED=DATASET_ROOT/'processed'
def pick_device():
    if os.environ.get('COCD_DEVICE'):return torch.device(os.environ['COCD_DEVICE'])
    if torch.cuda.is_available():return torch.device('cuda')
    if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available():return torch.device('mps')
    return torch.device('cpu')
DEV=pick_device()
