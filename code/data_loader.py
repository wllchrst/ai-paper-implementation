import torchvision

from torchvision.transforms import transforms
from torch.utils.data import DataLoader

def load_MNIST_dataset() -> dict[str, DataLoader]:
    transform = transforms.Compose\
        (transforms=[transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    training_dataset = torchvision.datasets.\
        MNIST(root='../data', train=True, download=True, transform=transform)

    return {}