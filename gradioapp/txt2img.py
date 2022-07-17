from builtins import breakpoint
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange
from einops import rearrange
from tempfile import TemporaryDirectory
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


CONFIG = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval.yaml")
LOADED_MODELS = {}


def load_model(ckpt, verbose=False):
    """Loads the requested model into model cache"""
    if ckpt not in LOADED_MODELS:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(CONFIG.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        model.cuda()
        model.eval()
        LOADED_MODELS[ckpt] = model

    return LOADED_MODELS[ckpt]


def render_image(prompt):
    """Renders an image for the given prompt"""
    outdir = TemporaryDirectory()
    ddim_steps = 50
    # ddim_steps = 500
    # ddim_eta = 1.3
    ddim_eta = 0.0  # NOTE: using eta 1.3 produces divergence for small ddim_steps. Why?
    n_samples = 9
    # scale = 12.0
    scale = 5.0
    H = W = 256

    model = load_model("models/ldm/text2img-large/model.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    outpath = outdir.name

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if scale != 1.0:
                uc = model.get_learned_conditioning(n_samples * [""])
            c = model.get_learned_conditioning(n_samples * [prompt])
            shape = [4, H//8, W//8]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=c,
                                                batch_size=n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc,
                                                eta=ddim_eta)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

            for x_sample in x_samples_ddim:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                all_samples.append(Image.fromarray(x_sample.astype(np.uint8)))
                # Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                # base_count += 1
            # all_samples.append(x_samples_ddim)

    # # additionally, save as grid
    # grid = torch.stack(all_samples, 0)
    # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    # grid = make_grid(grid, nrow=n_samples)

    # # to image
    # grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    # # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.png'))
    # return Image.fromarray(grid.astype(np.uint8))

    return tuple(all_samples)
    