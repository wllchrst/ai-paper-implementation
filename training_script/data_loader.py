import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from typing import Tuple

def show_image_example(image : torch.Tensor, label: str) -> None:
    """
    Show image from the dataset that is loaded
    Args:
        -
    Returns:
        - None: function will receive any input
    """
    image = image.permute(1, 2, 0).numpy()

    plt.imshow(image)
    plt.title(f"Image example from the dataset {label}")
    plt.axis('off')
    plt.show()

def load_cifar10_dataset(batch_size=8, show_image=True) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST Dataset from torchvision library
    Args:
    - batch_size: int Batch size for the loader
    Returns:
        Tuple[DataLoader, DataLoader]: Train Loader and Test Loader
    """
    if show_image:
        dataset = torchvision.datasets.CIFAR10(
            root="../data",
            transform=transforms.ToTensor(),
            download=True,
            train=False
        )

        image, label = dataset[0]
        show_image_example(image, label)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(224,224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3 channels mean/std
    ])

    training_dataset = torchvision.datasets.\
        CIFAR10(root='../data', train=True, download=True, transform=transform)

    test_dataset = torchvision.datasets.\
        CIFAR10(root='../data', train=False, download=True, transform=transform)

    train_loader = DataLoader\
        (training_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader\
        (test_dataset, batch_size=batch_size, shuffle=True)
        
    return train_loader, test_loader
