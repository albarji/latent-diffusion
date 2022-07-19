from functools import partial
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange
from tempfile import TemporaryDirectory

from ldm.conditioners.reference_image import reference_conditioner, encode_reference_image
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


def render_image(prompt, ddim_steps, scale, ref_image=None, ref_strength=0.0):
    """Renders an image for the given prompt"""
    outdir = TemporaryDirectory()
    # ddim_eta = 1.3
    ddim_eta = 0.0  # NOTE: using eta 1.3 produces divergence for small ddim_steps. Why?
    n_samples = 8
    H = W = 256

    model = load_model("models/ldm/text2img-large/model.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    # Prepare conditioners
    conditioner = None
    if ref_strength:
        latent_ref = encode_reference_image(model, ref_image)
        conditioner = partial(reference_conditioner, latent_ref=latent_ref, strength=ref_strength)

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
                                                eta=ddim_eta,
                                                cond_fn=conditioner)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

            for x_sample in x_samples_ddim:
                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                all_samples.append(Image.fromarray(x_sample.astype(np.uint8)))

    return tuple(all_samples)
    