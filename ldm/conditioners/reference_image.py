# Conditioners for a given image

from einops import rearrange
import torch


def blank_conditioner(x, t, strength):
    """Conditions the image to produce blank (all black pixel) images (just for testing)"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    x = x.detach()
    blank = torch.zeros(x.shape).to(device)
    with torch.enable_grad():
        x = x.requires_grad_()
        loss = strength * (x - blank).pow(2).mean()
        loss.backward()
        grad = -x.grad
    return grad


def reference_conditioner(x, t, latent_ref, strength):
    """Conditions the image to produce follow the given reference image in the pixel space"""
    # Encode reference image into the latent space
    x = x.detach()
    with torch.enable_grad():
        x = x.requires_grad_()
        loss = strength * (latent_ref - x).pow(2).mean()
        loss.backward()
        grad = -x.grad
    return grad


def encode_reference_image(model, ref_image):
    """Encodes a reference image into the latent space"""
    ref_image = torch.tensor(ref_image)
    ref_image = rearrange(ref_image, 'h w c -> 1 c h w').repeat([8, 1, 1, 1])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ref_image = ref_image.to(device)
    # latent_ref = model.encode_first_stage(ref_image / 255.0 * 2.0 - 1.0).mean  # TODO: check correct normalization
    latent_ref = model.encode_first_stage(ref_image.to(torch.float32) / 255.0).mean  # TODO: check correct normalization
    return latent_ref
