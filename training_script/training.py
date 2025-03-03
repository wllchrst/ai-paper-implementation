import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def test_model(
    model: nn.Module,
    test_loader: DataLoader
    ) -> None:
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {correct / total}')

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs=3,
    with_test=True
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

        print(f"Epoch: {epoch}, Loss: {loss.item()}")

    if with_test:
        test_model(model, test_loader)