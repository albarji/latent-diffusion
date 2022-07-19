# Dummy conditioners for testing

import torch


def blank_conditioner(x, t, strength):
    """Conditions the image to produce blank (all black pixel) images"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    x = x.detach()
    blank = torch.zeros(x.shape).to(device)
    with torch.enable_grad():
        x = x.requires_grad_()
        loss = strength * (x - blank).pow(2).mean()
        loss.backward()
        grad = -x.grad
    return grad
