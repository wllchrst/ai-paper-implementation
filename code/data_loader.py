import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from typing import Tuple

def load_MNIST_dataset(batch_size=8) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST Dataset from torchvision library

    Args:
    - batch_size: int Batch size for the loader

    Returns:
        Tuple[DataLoader, DataLoader]: Train Loader and Test Loader
    """
    transform = transforms.Compose\
        (transforms=[transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    training_dataset = torchvision.datasets.\
        MNIST(root='../data', train=True, download=True, transform=transform)

    test_dataset = torchvision.datasets.\
        MNIST(root='../data', train=False, download=True, transform=transform)

    train_loader = DataLoader\
        (training_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader\
        (test_dataset, batch_size=batch_size, shuffle=True)
        
    return train_loader, test_loader
