import torch

from torch import nn

from torch.utils.data import DataLoader, TensorDataset

import torch.optim as optim

from einops import rearrange
from tqdm import tqdm

from dataclasses import dataclass
from generate_mass_spring_data import generate_mass_spring_data
from loss import compute_mmd

@dataclass
class TrainConfig:
    num_epochs: int
    batch_size: int
    learning_rate: float
    device: str
    model: torch.nn.Module
    optimizer: optim.Optimizer 
    train_loader: DataLoader
    val_loader: DataLoader

def get_data_loaders(n_points: int):
    output = generate_mass_spring_data(torch.tensor(1), torch.tensor(1), 1, 0, 0.1, n_points)
    output = rearrange(torch.stack(output), 'c b -> b c')
    dataset = TensorDataset(output)

    train_ds, val_ds = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])

    return DataLoader(train_ds, batch_size=32), DataLoader(val_ds, batch_size=32)
    
    


def train(config: TrainConfig):
    losses = []
    for epoch in range(config.num_epochs):
        ep_losses = []
        config.model.train()
        for i, data in enumerate(config.train_loader):
            data = data[0].to(config.device)
            yhat = config.model(data)
            loss = compute_mmd(data, yhat)
            config.optimizer.zero_grad()
            loss.backward()
            ep_losses.append(loss.clone().item())
            config.optimizer.step()
        config.model.eval()
        losses.append(sum(ep_losses)/len(ep_losses))
        print(f'Epoch {epoch+1}/{config.num_epochs}, Loss: {loss.item()}')
    return config.model

if __name__ == '__main__':
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )

    train_loader, val_loader = get_data_loaders(1000)

    config = TrainConfig(
        num_epochs=100,
        batch_size=32,
        learning_rate=0.001,
        device='cpu',
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        train_loader=train_loader,
        val_loader=val_loader
    )

    model = train(config)
