from torchvision import datasets, transforms
from torchvision import transforms
import torch

def get_mnist_dataloaders(batch_size=2):
    # Source: https://gist.github.com/kdubovikov/eb2a4c3ecadd5295f68c126542e59f0a
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            download=True,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            download=True,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )
    return train_loader, test_loader
