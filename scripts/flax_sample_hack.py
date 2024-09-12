import jax
import jax.numpy as jnp
import jax.nn as jnn
import flax
from flax.linen.linear import Array
from flax import linen as nn
from flax.serialization import to_bytes, from_bytes
from typing import Any, Callable, Dict, Optional, Tuple, Union
import optax
from flax.training import train_state
import pickle

import argparse
import os
import tqdm
from tools import file_utils
from tools import cv2_utils
import logging
import time
import json
import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2
import spu.utils.distributed as ppd
from contextlib import contextmanager

import imageio
from flax_ddpm.script_utils import get_args
from flax_ddpm.script_utils import get_diffusion_from_args

copts = spu_pb2.CompilerOptions()
copts.enable_pretty_print = False
copts.xla_pp_kind = 2
# enable x / broadcast(y) -> x * broadcast(1/y)
copts.enable_optimize_denominator_with_broadcast = True

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='distributed driver.')
parser.add_argument("-c", "--config", default="3pc.json")
args = parser.parse_args()

with open(args.config, 'r') as file:
    conf = json.load(file)
ppd.init(conf["nodes"], conf["devices"])



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



def transform_x(x, x_min=-14, x_max=0):
    return 2 * (x - x_min) / (x_max - x_min) - 1

def chexp(x):
    x = transform_x(x)
    x2 = x*x
    x3 = x2*x
    x4 = x3*x
    x5 = x4*x
    x6 = x5*x
    x7 = x6*x
    t0 = 1
    t1 = x
    t2 = 2*x2 -1
    t3 = 4*x3 -3*x
    t4 = 8*x4 -8*x2 +1 
    t5 = 16*x5 - 20*x3 +5*x
    t6 = 32*x6 - 48*x4 +18*x2 -1
    t7 = 64*x7 -112*x5 +56*x3 -7*x
    ex= 0.14021878*t0 + 0.27541278*t1+ 0.22122865*t2 + 0.14934221*t3 + 0.0907736*t4 + 0.04369614*t5 + 0.02087868*t6 + 0.00996535*t7
    return ex

def hack_softmax(x: Array,
            axis: Optional[Union[int, Tuple[int, ...]]] = -1,
            where: Optional[Array] = None,
            initial: Optional[Array] = None) -> Array:

    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max

    # exp on large negative is clipped to zero
    b = x > -14
    nexp = chexp(x) * b

    divisor = jnp.sum(nexp, axis, where=where, keepdims=True)

    return nexp / divisor

@contextmanager
def hack_softmax_context(msg: str, enabled: bool = False):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_softmax = nn.softmax
    nn.softmax = hack_softmax
    yield
    # recover back
    nn.softmax = raw_softmax


def hack_silu(x: Array) -> Array:
    b0 = x < -6.0
    b1 = x < -2.0
    b2 = x > 6.0
    b3 = b1 ^ b2 ^ True  # x in [-4.0, 4.0)
    b4 = b0 ^ b1  # x in [-8.0, -4.0)
    a_coeffs = jnp.array(
        [-0.52212664, -0.16910363, -0.01420163]
    )
    b_coeffs = jnp.array(
        [
            0.03453821,
            0.49379432,
            0.19784596,
            -0.00602401,
            0.00008032,
        ]
    )
    x2 = jnp.square(x)
    x4 = jnp.square(x2)
    x6 = x2 * x4
    seg1 = a_coeffs[2] * x2 + a_coeffs[1] * x + a_coeffs[0]
    seg2 = (
        b_coeffs[4] * x6
        + b_coeffs[3] * x4
        + b_coeffs[2] * x2
        + b_coeffs[1] * x
        + b_coeffs[0]
    )
    ret = b2 * x + b4 * seg1 + b3 * seg2
    return ret

@contextmanager
def hack_silu_context(msg: str, enabled: bool = True):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_silu = nn.silu
    nn.silu = hack_silu
    yield
    # recover back
    nn.silu = raw_silu

def hack_mish(x: Array) -> Array:
    b0 = x < -6.0
    b1 = x < -2.0
    b2 = x > 6.0
    b3 = b1 ^ b2 ^ True  # x in [-4.0, 4.0)
    b4 = b0 ^ b1  # x in [-8.0, -4.0)
    a_coeffs = jnp.array(
        [-0.55684445, -0.18375535, -0.01572019]
    )
    b_coeffs = jnp.array(
        [
            0.07559242,
            0.54902050,
            0.20152583,
            -0.00735309,
            0.00010786,
        ]
    )
    x2 = jnp.square(x)
    x4 = jnp.square(x2)
    x6 = x2 * x4
    seg1 = a_coeffs[2] * x2 + a_coeffs[1] * x + a_coeffs[0]
    seg2 = (
        b_coeffs[4] * x6
        + b_coeffs[3] * x4
        + b_coeffs[2] * x2
        + b_coeffs[1] * x
        + b_coeffs[0]
    )
    ret = b2 * x + b4 * seg1 + b3 * seg2
    return ret


@contextmanager
def hack_mish_context(msg: str, enabled: bool = True):
    if not enabled:
        yield
        return
    # hijack some target functions
    raw_mish = mish
    mish = hack_mish
    yield
    # recover back
    mish = raw_mish

@measure_runtime
def run_inference_on_cpu(params,y,save_dir):
    output = sample_image(params,y)
    # imageio.imwrite(f"{save_dir}/raw.png", output)


@measure_runtime
def run_inference_on_spu(params,y,save_dir):
    output = ppd.device("SPU")(sample_image, copts=copts)(params, y)
    output = ppd.get(output)
    # imageio.imwrite(f"{save_dir}/safe.png", output)
    
@measure_runtime
def run_inference_on_hackspu(params,y,save_dir):
    output = ppd.device("SPU")(sample_image, copts=copts)(params, y)
    output = ppd.get(output)
    # imageio.imwrite(f"{save_dir}/hacksafe.png", output)

def get_sample_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str, default="./model_eval")
    parser.add_argument("--num_samples", type=int, default=10)
    return parser


def normalize_to_0_1(array):
    min_val = array.min()
    max_val = array.max()
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

def sample_image(params, y=None):
    diffusion = get_diffusion_from_args(args)
    rng = jax.random.PRNGKey(0)  
    # samples = diffusion.apply({'params': params}, batch_size=1, y=y, use_ema= False, rng_key=rng, method=diffusion.sample)
    samples = diffusion.apply({'params': params}, batch_size=1, y=y, use_ema= False, rng_key=rng, method=diffusion.sample_ddim)
    # Process the image
    image = normalize_to_0_1(samples[0])
    image = (image * 255).astype(jnp.uint8)  # Convert to [0, 255]
    return image


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
    x = jax.random.normal(rng, (128,28,28,1))
    y = jnp.ones(128, dtype=jnp.int32)
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

    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.info(f"===== Starting Testing=====")
    

    #public evaluation
    params = restored_state.params
    y = jnp.ones(num_each_label, dtype=jnp.int32)
    run_inference_on_cpu(params,y,save_dir)

    logger.info(f"Public Eval Completed! Start Secure Eval:----")   
    y_secret = ppd.device("INPUT")(lambda x: x)(y)
    params_secret = ppd.device("INPUT")(lambda x: x)(params)
    run_inference_on_spu(params_secret,y_secret,save_dir)
    
    # Puma evaluation
    logger.info(f"Public Eval Completed! Start Secure Eval:hack") 
    with hack_softmax_context("hijack jax softmax"), hack_silu_context("hack jax silu"):
        y_secret = ppd.device("INPUT")(lambda x: x)(y)
        params_secret = ppd.device("INPUT")(lambda x: x)(params)
        run_inference_on_hackspu(params_secret,y_secret,save_dir)
    print("end")




if __name__ == "__main__":
    parser = get_sample_arg_parser()
    args = get_args(parser)
    for k, v in args.__dict__.items():
        print(f"===> {k}: {v}")
    main(args)
