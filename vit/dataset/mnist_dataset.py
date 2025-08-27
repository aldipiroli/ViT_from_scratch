import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets import VOCDetection
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        self.root_dir = Path(cfg["DATA"]["root_dir"])
        dest_path = os.path.join(self.root_dir)
        self.dataset = datasets.MNIST(
            dest_path,
            download=False,
            train=True if mode == "train" else False,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        return img, target
