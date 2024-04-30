import torch
import numpy as np

from torch import nn

from torch.utils.data import DataLoader, TensorDataset

import torch.optim as optim

from einops import rearrange
from tqdm import tqdm

from dataclasses import dataclass
from generate_mass_spring_data import generate_mass_spring_data
from loss import compute_mmd

from matplotlib import pyplot as plt

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
    output = generate_mass_spring_data(torch.tensor(1), torch.tensor(1), 1, 1, 0.1, n_points)
    output = rearrange(torch.stack(output), 'c b -> b c')
    dataset = TensorDataset(output)

    train_ds, val_ds = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])

    return DataLoader(train_ds, batch_size=32), DataLoader(val_ds, batch_size=32)
    
    


def train(config: TrainConfig):
    losses = []
    conservation_losses = []
    reconstruction_losses = []
    for epoch in range(config.num_epochs):
        ep_losses = []
        ep_conservation_losses = []
        ep_reconstruction_losses = []
        config.model.train()

        for i, data in enumerate(config.train_loader):
            data = data[0].to(config.device)
            encoder, decoder = config.model
            z = encoder(data)
            yhat = decoder(z)

            conservation_loss = (torch.sum(z)/z.size(0) - torch.mean(z))**2

            reconstruction_loss = torch.mean((data - yhat)**2)

            loss = reconstruction_loss + conservation_loss
            # loss = compute_mmd(data, yhat)
            config.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            ep_losses.append(loss.clone().item())
            ep_conservation_losses.append(conservation_loss.clone().item())
            ep_reconstruction_losses.append(reconstruction_loss.clone().item())
            config.optimizer.step()
        config.model.eval()
        losses.append(np.mean(ep_losses))
        conservation_losses.append(np.mean(ep_conservation_losses))
        reconstruction_losses.append(np.mean(ep_reconstruction_losses))
        print(f'Epoch {epoch+1}/{config.num_epochs}, Loss: {loss.item()}')
    return config.model, losses, conservation_losses, reconstruction_losses

if __name__ == '__main__':
    encoder = nn.Sequential(
        nn.Linear(2, 1),
        nn.ReLU(),
        nn.Linear(1, 1)
    )

    decoder = nn.Sequential(
        nn.Linear(1, 1),
        nn.ReLU(),
        nn.Linear(1, 2)
    )

    model = nn.Sequential(encoder, decoder)

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

    model, loss, conservation_loss, reconstruction_loss = train(config)
    model = model.eval()
    x_test = torch.ones((100,1))
    v_test = rearrange(torch.linspace(1, 1.5, 100), 'b -> b 1')
    test = rearrange(torch.stack((x_test, v_test)), 'c b a -> b (c a)')
    zs = model[0](test)
    plt.plot(v_test, zs.detach().numpy(), label='z')
    #set y limit
    plt.ylim(-1, 1)
    plt.legend()
    plt.show()

    # datapoint = val_loader.dataset[0][0]
    # energy_gnd_th = 0.5 * 1 * datapoint[0]**2 + 0.5 * 1 * datapoint[1]**2
    # energy_pred = model[0](datapoint).item()
    # print(f'Ground truth energy: {energy_gnd_th}, Predicted energy: {energy_pred}')
    #
    # plt.plot(loss, label='Total loss')
    # plt.plot(conservation_loss, label='Conservation loss')
    # plt.plot(reconstruction_loss, label='Reconstruction loss')
    # plt.legend()
    # plt.show()
