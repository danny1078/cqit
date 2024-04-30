import torch

def generate_mass_spring_data(k, m, x0, v0, dt, num_steps):
    w = torch.sqrt(k / m)
    t = torch.arange(num_steps).float() * dt
    x = x0 * torch.cos(w * t) + v0 / w * torch.sin(w * t)
    v = -x0 * w * torch.sin(w * t) + v0 * torch.cos(w * t)
    return x, v
