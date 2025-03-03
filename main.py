import torch
import warnings
from training_script import data_loader as dl 
from training_script import training as t
from alex_net import model as alex_net

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will use device: {device}")

    train_loader, test_loader = dl.load_cifar10_dataset(show_image=False)
    model = alex_net.AlexNet(3).to(device)

    t.train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
       num_epochs=3
    )
