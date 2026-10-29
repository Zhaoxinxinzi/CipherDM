"""Map original nanoDiffusion UNet weights to the aligned Flax parameter tree."""
import numpy as np
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict


def torch_name(path):
    """Return the original Torch named_parameter for a Flax leaf path."""
    parts = list(path)
    if parts[0] == 'model':
        parts.pop(0)
    leaf = parts.pop()
    suffix = 'weight' if leaf in ('kernel', 'scale', 'embedding') else leaf
    top = parts.pop(0)
    if top == 'time_mlp':
        return f'time_mlp.{parts[0].split("_")[-1]}.{suffix}'
    if top.startswith(('downs_', 'mid_', 'ups_')):
        group, index = top.rsplit('_', 1)
        prefix = f'{group}.{index}'
        layer = parts.pop(0)
        if layer == 'AttentionBlock_0':
            return f'{prefix}.attention.{parts[0]}.{suffix}'
        names = {'GroupNorm_0': 'norm_1', 'GroupNorm_1': 'norm_2',
                 'BatchNorm_0': 'norm_1', 'BatchNorm_1': 'norm_2',
                 'Dense_0': 'time_bias', 'Embed_0': 'class_bias',
                 'Conv_0': 'conv_1', 'Conv_1': 'conv_2.1',
                 'Conv_2': 'residual_connection'}
        # A down/upsampling module consists only of one convolution.
        return f'{prefix}.{names[layer]}.{suffix}'
    return f'{top}.{suffix}'


def convert_parameters(weights, template):
    """Convert a Torch named-parameter mapping (NumPy arrays); reject mismatches.

    The template distinguishes a sampling layer from a residual block. Conv2d
    OIHW -> HWIO; Linear OI -> IO; normalization and embedding arrays stay intact.
    """
    flat = flatten_dict(template)
    block_parents = {path[:-2] for path in flat if path[-2] == 'Conv_1'}
    converted, used = {}, set()
    for path, target in flat.items():
        name = torch_name(path)
        parent = path[:-2]
        block = parent[-1] if parent else ''
        if block.startswith(('downs_', 'ups_')) and parent not in block_parents:
            group, index = block.rsplit('_', 1)
            operator = 'downsample' if group == 'downs' else 'upsample.1'
            name = f'{group}.{index}.{operator}.{path[-1] if path[-1] == "bias" else "weight"}'
        if name not in weights:
            raise ValueError(f'Missing Torch parameter {name} for {"/".join(path)}')
        array = np.asarray(weights[name])
        if path[-1] == 'kernel':
            array = array.transpose(2, 3, 1, 0) if array.ndim == 4 else array.T
        if array.shape != target.shape:
            raise ValueError(f'Shape mismatch for {name}: {array.shape} != {target.shape}')
        converted[path] = jnp.asarray(array, dtype=target.dtype)
        used.add(name)
    unused = set(weights) - used
    if unused:
        raise ValueError(f'Unused Torch parameters: {sorted(unused)}')
    return unflatten_dict(converted)
