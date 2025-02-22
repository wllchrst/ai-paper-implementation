import torch
from data_loader import load_MNIST_dataset 
from model import AlexNet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will use device: {device}")

    train_loader, test_loader = load_MNIST_dataset()
    model = AlexNet(1).to(device)
