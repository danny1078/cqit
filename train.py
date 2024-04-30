import torch

from torch import nn

from torch.utils.data import DataLoader, TensorDataset

import torch.optim as optim

from dataclasses import dataclass
from generate_mass_spring_data import generate_mass_spring_data

@dataclass
class TrainConfig:
    num_epochs: int
    batch_size: int
    learning_rate: float
    device: str
    model: torch.nn.Module
    train_loader: DataLoader
    val_loader: DataLoader

def get_data_loaders(n_points: int):
    output = generate_mass_spring_data(torch.tensor(1), torch.tensor(1), 1, 0, 0.1, n_points)
    output = torch.stack(output)
    # breakpoint()

    dataset = TensorDataset(output)
    # random split in torch
    train_ds, val_ds = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])

    return DataLoader(train_ds, batch_size=32), DataLoader(val_ds, batch_size=32)
    
    


def train(config: TrainConfig):
    for epoch in range(config.num_epochs):
        config.model.train()
        for i, data in enumerate(config.train_loader):
            yhat = config.model(data)
            loss = config.model.loss(outputs, labels)
            config.optimizer.zero_grad()
            loss.backward()
            config.optimizer.step()
        config.model.eval()
        # with torch.no_grad():

        print(f'Epoch {epoch+1}/{config.num_epochs}, Loss: {loss.item()}')
        if (epoch+1) % 10 == 0:
            torch.save(config.model.state_dict(), f'./model_{epoch+1}.pth')
        torch.save(config.model.state_dict(), './model.pth')
        return config.model

if __name__ == '__main__':

    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )
    train_loader, val_loader = get_data_loaders()
    config = TrainConfig(
        num_epochs=100,
        batch_size=32,
        learning_rate=0.001,
        device='cpu',
        model=model,
        train_loader=train_loader,
        val_loader=val_loader
    )
    model = train(config)
