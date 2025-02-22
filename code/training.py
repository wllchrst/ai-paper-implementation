import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs=3,
    ) -> None:

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for batch in train_loader:
            images, labels = batch
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()


    for batch in test_loader:
        