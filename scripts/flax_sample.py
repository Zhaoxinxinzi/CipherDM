"""Sample an aligned Flax checkpoint; DDPM matches the original Torch default."""
import argparse
from pathlib import Path
import pickle

import flax
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from flax_ddpm.script_utils import get_args, get_diffusion_from_args


def get_sample_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--save_dir', default='./model_eval')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--label', type=int, default=1)
    parser.add_argument('--sampler', choices=['ddpm', 'ddim'], default='ddpm')
    parser.add_argument('--ddim_timesteps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_ema', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--no_use_ema', dest='use_ema', action='store_false')
    return parser


def main(args):
    if args.num_samples < 1 or not 0 <= args.label < args.num_classes:
        raise ValueError('num_samples must be positive and label must be in the class range')
    with open(args.model_path, 'rb') as handle:
        checkpoint = pickle.load(handle)
    weights = checkpoint.get('ema_params', checkpoint['params']) if args.use_ema else checkpoint['params']
    if any('ConvTranspose' in '/'.join(path) for path in flatten_dict(weights)):
        raise ValueError('Legacy ConvTranspose checkpoint: retrain or convert Torch weights for the aligned UNet')
    diffusion = get_diffusion_from_args(args)
    labels = jnp.full((args.num_samples,), args.label, dtype=jnp.int32)
    key = jax.random.PRNGKey(args.seed)
    initial = jnp.zeros((args.num_samples, *args.img_size, args.img_channels))
    template = diffusion.init(key, key, initial, labels)['params']
    params = flax.serialization.from_state_dict(template, weights)
    actual_flat, expected_flat = flatten_dict(params), flatten_dict(template)
    for path, expected in expected_flat.items():
        if actual_flat[path].shape != expected.shape:
            raise ValueError(f'Checkpoint shape mismatch at {path}; use matching architecture arguments')
    kwargs = dict(batch_size=args.num_samples, y=labels, use_ema=False, rng_key=key)
    if args.sampler == 'ddim':
        kwargs['ddim_timesteps'] = args.ddim_timesteps
        samples = diffusion.apply({'params': params}, method=diffusion.sample_ddim, **kwargs)
    else:
        samples = diffusion.apply({'params': params}, method=diffusion.sample, **kwargs)
    samples = np.asarray(samples)
    if not np.isfinite(samples).all():
        raise FloatingPointError('Sampler produced non-finite values')
    images = np.uint8(np.clip((samples+1)*127.5, 0, 255))
    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        Image.fromarray(image[..., 0] if image.shape[-1] == 1 else image).save(out/f'{args.label}_{index}.png')
    ncol = min(5, len(images))
    nrow = (len(images)+ncol-1)//ncol
    height, width, channels = images.shape[1:]
    grid = np.zeros((nrow*height, ncol*width, channels), dtype=np.uint8)
    for index, image in enumerate(images):
        row, col = divmod(index, ncol)
        grid[row*height:(row+1)*height, col*width:(col+1)*width] = image
    Image.fromarray(grid[..., 0] if channels == 1 else grid).save(out/f'label_{args.label}_grid.png')
    print(f'Saved {len(images)} {args.sampler} samples to {out}', flush=True)


if __name__ == '__main__':
    main(get_args(get_sample_arg_parser()))
