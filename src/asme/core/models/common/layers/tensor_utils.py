import torch


def generate_position_ids(shape,
                          device: torch.device = None):
    position_ids = torch.arange(shape[1], dtype=torch.long, device=device)
    return position_ids.unsqueeze(0).repeat(shape[0], 1)
