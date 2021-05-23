import torch 
def weighted_loss(output, target):
    diff = torch.abs(output - target)/target
    # mask_t = torch.where(target >= 1, 100, 1)
    # mask_o = torch.where(output >= 1, 100, 1)
    # mask = mask_t * mask_o
    mask = torch.where(diff >= 0.1, 10, 1)
    loss = torch.mean(mask*torch.abs(output - target))
    return loss