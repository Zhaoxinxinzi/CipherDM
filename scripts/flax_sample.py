import jax
import jax.numpy as jnp
import flax
import optax
from flax.training import train_state
from flax import struct
import pickle

import argparse
import os
import imageio
import numpy as np
from tools import file_utils
from flax_ddpm.script_utils import get_args
from flax_ddpm.script_utils import get_diffusion_from_args
import time


@struct.dataclass
class EMATrainState(train_state.TrainState):
    ema_params: object



def get_sample_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./model_eval')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--label', type=int, default=1)
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--no_use_ema', dest='use_ema', action='store_false')
    return parser



def measure_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Function '{func.__name__}' took {runtime:.4f} seconds to complete.")
        return result
    return wrapper



def denormalize_to_uint8(array):
    array = ((array + 1.0) * 127.5).clip(0, 255)
    return array.astype(jnp.uint8)


@measure_runtime
def sample_images(diffusion, params, labels, args):
    rng = jax.random.PRNGKey(0)
    samples = diffusion.apply(
        {'params': params},
        batch_size=args.num_samples,
        y=labels,
        use_ema=False,
        rng_key=rng,
        method=diffusion.sample_ddim,
    )
    return denormalize_to_uint8(samples)



def make_grid(images, nrow):
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[..., None]
    n, h, w, c = images.shape
    ncol = min(nrow, n)
    nrows = (n + ncol - 1) // ncol
    grid = np.zeros((nrows * h, ncol * w, c), dtype=images.dtype)
    for idx, image in enumerate(images):
        row = idx // ncol
        col = idx % ncol
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w, :] = image
    return grid



def main(args: argparse.Namespace):
    model_path = args.model_path
    save_dir = args.save_dir

    assert os.path.exists(model_path), f'model file not exist: {model_path}'
    file_utils.mkdir(save_dir)

    batch_size = args.num_samples
    rng = jax.random.PRNGKey(0)
    diffusion = get_diffusion_from_args(args)
    x = jax.random.normal(rng, (batch_size, 28, 28, 1))
    y = jnp.ones(batch_size, dtype=jnp.int32) * args.label
    variables = diffusion.init(rngs=rng, rng=rng, x=x, y=y)
    optimizer = optax.adam(learning_rate=args.learning_rate)

    state = EMATrainState.create(
        apply_fn=diffusion.apply,
        params=variables['params'],
        tx=optimizer,
        ema_params=variables['params'],
    )

    with open(model_path, 'rb') as f:
        ckpt = pickle.load(f)

    restored_state = state.replace(
        step=jnp.array(ckpt['step']),
        params=flax.serialization.from_state_dict(state.params, ckpt['params']),
        ema_params=flax.serialization.from_state_dict(state.ema_params, ckpt.get('ema_params', ckpt['params'])),
        opt_state=flax.serialization.from_state_dict(state.opt_state, ckpt['opt_state']),
    )

    sample_params = restored_state.ema_params if args.use_ema else restored_state.params
    labels = jnp.ones(args.num_samples, dtype=jnp.int32) * args.label
    images = sample_images(diffusion, sample_params, labels, args)

    for idx, image in enumerate(np.asarray(images)):
        out = image.squeeze(-1) if image.shape[-1] == 1 else image
        imageio.imwrite(os.path.join(save_dir, f'{args.label}_{idx}.png'), out)

    grid = make_grid(images, nrow=min(5, args.num_samples))
    grid = grid.squeeze(-1) if grid.shape[-1] == 1 else grid
    imageio.imwrite(os.path.join(save_dir, f'label_{args.label}_grid.png'), grid)
    print(f'Saved {args.num_samples} sample(s) to {save_dir}')


if __name__ == '__main__':
    parser = get_sample_arg_parser()
    args = get_args(parser)
    for k, v in args.__dict__.items():
        print(f'===> {k}: {v}')
    main(args)
