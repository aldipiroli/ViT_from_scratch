import sys
import os
import torch
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import divide_in_patches
from models.cnn_vit import ImagePatcher

def test_divide_in_patches():
    patch_size = 4
    x = torch.rand(2, 28, 28, 1)
    x_patch_manual = divide_in_patches(x, patch_size)
    cnn_patcher = ImagePatcher(in_size=1, out_size=patch_size**2, patch_size=patch_size)
    x = torch.rand(2, 1, 28, 28)
    x_patch_cnn = cnn_patcher(x)
    assert x_patch_manual.shape == x_patch_cnn.shape
