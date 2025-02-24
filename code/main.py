import torch
import warnings
from training import train_model
from data_loader import load_cifar10_dataset
from model import AlexNet

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will use device: {device}")

    train_loader, test_loader = load_cifar10_dataset(show_image=False)
    model = AlexNet(3).to(device)

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
       num_epochs=3
    )