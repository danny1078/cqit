import torch
def energy_angular_momentum(x_0, v_0, m, k):
    energy = 0.5 * m * torch.sum(v_0 ** 2, dim=1) + 0.5 * k * torch.sum(x_0 ** 2, dim=1)
    energy = energy.unsqueeze(1)
    angular_momentum = torch.linalg.cross(x_0, m * v_0)
    return energy, angular_momentum