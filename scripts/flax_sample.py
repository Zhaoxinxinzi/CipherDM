import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax
from flax.linen.linear import Array
from flax import linen as nn
import optax
from flax.training import train_state
import pickle

import argparse
import os
import tqdm
from tools import file_utils
from tools import cv2_utils
import time

import imageio
from flax_ddpm.script_utils import get_args
from flax_ddpm.script_utils import get_diffusion_from_args
import sys

def get_sample_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str, default="./model_eval")
    parser.add_argument("--num_samples", type=int, default=10)
    return parser


def measure_runtime(func):
    """
    A decorator to measure the runtime of a function.

    Parameters:
    func (callable): The function whose runtime you want to measure.

    Returns:
    callable: A wrapper function that will measure the runtime of 'func'.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  
        result = func(*args, **kwargs)  
        end_time = time.time()  
        runtime = end_time - start_time  
        print(f"Function '{func.__name__}' took {runtime:.4f} seconds to complete.")
        return result
    return wrapper


def normalize_to_0_255(array):
    min_val = array.min()
    max_val = array.max()
    normalized_array = (array - min_val) *255/ (max_val - min_val)
    return normalized_array.astype(jnp.uint8)

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

@measure_runtime
def sample_image(params, y=None):
    diffusion = get_diffusion_from_args(args)
    rng = jax.random.PRNGKey(0) 
    rng, rng_params = jax.random.split(rng)
    samples = diffusion.apply({'params': params}, batch_size=1, y=y, use_ema= False, rng_key=rng, method=diffusion.sample_ddim)
    # samples = diffusion.apply({'params': params}, batch_size=1, y=y, use_ema= False, rng_key=rng, method=diffusion.sample)
    image = normalize_to_0_255(samples[0])
    return image


import jax.numpy as jnp

def make_grid(images, nrow):
    if len(images) == 0:
        raise ValueError("No images to display in the grid.")
    single_image_shape = images[0].shape
    if len(single_image_shape) == 2:  
        height, width = single_image_shape
        num_channels = 1
    elif len(single_image_shape) == 3:  
        height, width, num_channels = single_image_shape
    else:
        raise ValueError("Images should be 2D or 3D.")
    nrows = (len(images) + nrow - 1) // nrow
    grid_height = nrows * single_image_shape[0]
    grid_width = nrow * single_image_shape[1]
    grid = jnp.zeros((grid_height, grid_width, num_channels), dtype=images[0].dtype)
    for index, image in enumerate(images):
        row = index // nrow
        col = index % nrow
        image_view = image.reshape(height, width, num_channels) if num_channels == 1 else image
        start_row, start_col = row * height, col * width
        end_row, end_col = (row + 1) * height, (col + 1) * width
        grid = grid.at[start_row:end_row, start_col:end_col, :].set(image_view)
    return grid


@measure_runtime
def sample_sequence(params, y=None):
    save_dir = "./eval_out"
    label=1
    diffusion = get_diffusion_from_args(args)
    rng = jax.random.PRNGKey(0) 
    def generate_images() -> "yield image numpy array":
        gen = diffusion.apply({'params': params}, batch_size=1, y=y, use_ema= False, rng_key=rng, method=diffusion.sample_sequence)
        for idx, image_tensor in tqdm.tqdm(enumerate(gen), desc=f"Generating for label {label}..", total=args.num_timesteps):
            if idx % 5 != 0:  # 1000 / 5 = 200 frames
                continue
            grid = make_grid(image_tensor, nrow=1)
            arr = (grid * 255 + 0.5).clip(0, 255).astype(jnp.uint8)
            image_bgr = arr[..., ::-1]
            yield image_bgr

    to_gif = os.path.join(save_dir, f"{label}.gif")
    cv2_utils.images_to_gif(list(generate_images()), to_gif)

def main(args: argparse.Namespace):
    # key args
    model_path = "./checkpoints/ddpm-20000.msgpack"
    save_dir = "./eval_out"
     # Load the model checkpoint

    assert model_path is not None
    assert os.path.exists(model_path), f"model file not exist: {model_path}"
    file_utils.mkdir(save_dir)
    num_each_label = 1
    batch_size =1

    #init
    rng = jax.random.PRNGKey(0)
    diffusion = get_diffusion_from_args(args)
    x = jax.random.normal(rng, (batch_size,28,28,1))
    y = jnp.ones(batch_size, dtype=jnp.int32)*1
    variables = diffusion.init(rngs=rng,rng=rng, x=x,y=y)# Initialize the parameters
    optimizer = optax.adam(learning_rate=args.learning_rate)


    state = train_state.TrainState.create(
            apply_fn=diffusion.apply,
            params=variables['params'],
            tx=optimizer,
        )

    # load params
    pkl_file=pickle.load(open(model_path,"rb"))
    restored_state=flax.serialization.from_state_dict(target=state, state=pkl_file)

    # sample
    params = restored_state.params
    y = jnp.ones(num_each_label, dtype=jnp.int32)*1
    image = sample_image(params,y)
    # imageio.imwrite(f"{save_dir}/1.png", image)



if __name__ == "__main__":
    parser = get_sample_arg_parser()
    args = get_args(parser)
    for k, v in args.__dict__.items():
        print(f"===> {k}: {v}")
    main(args)
