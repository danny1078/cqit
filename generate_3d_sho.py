import torch

from einops import rearrange

def generate_3d_sho(k, m, x0, v0, dt, num_steps):

    w_x = torch.sqrt(torch.tensor(k[0].item()/m))
    w_y = torch.sqrt(torch.tensor(k[1].item()/m))
    w_z = torch.sqrt(torch.tensor(k[2].item()/m))
    t = torch.arange(num_steps).float() * dt
    x = x0[0].item() * torch.cos(w_x * t) + v0[0].item() / w_x * torch.sin(w_x * t)
    y = x0[1].item() * torch.cos(w_y * t) + v0[1].item() / w_y * torch.sin(w_y * t)
    z = x0[2].item() * torch.cos(w_z * t) + v0[2].item() / w_z * torch.sin(w_z * t)
    v_x = -x0[0].item() * w_x * torch.sin(w_x * t) + v0[0].item() * torch.cos(w_x * t)
    v_y = -x0[1].item() * w_y * torch.sin(w_y * t) + v0[1].item() * torch.cos(w_y * t)
    v_z = -x0[2].item() * w_z * torch.sin(w_z * t) + v0[2].item() * torch.cos(w_z * t)

    pos = rearrange([x, y, z], 'd n -> n d')
    vel = rearrange([v_x, v_y, v_z], 'd n -> n d')
    return pos, vel