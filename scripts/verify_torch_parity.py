"""Verify layer outputs, gradients, Adam, and DDPM with shared Torch weights."""
import argparse
import json
from pathlib import Path
import pickle

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from PIL import Image

from flax_ddpm.script_utils import get_args, get_diffusion_from_args
from flax_ddpm.torch_compat import convert_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output_dir', required=True)
    args = get_args(parser)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reference = np.load(args.reference)
    diffusion = get_diffusion_from_args(args)
    model = diffusion.model
    nhwc = lambda array: jnp.asarray(array.transpose(0, 2, 3, 1))
    x, target = nhwc(reference['x']), nhwc(reference['target'])
    times, labels = jnp.asarray(reference['times']), jnp.asarray(reference['labels'])
    template = model.init(jax.random.PRNGKey(0), x, times, labels)['params']
    select = lambda prefix: {key[len(prefix):]: reference[key] for key in reference.files
                             if key.startswith(prefix)}
    params = convert_parameters(select('weight/'), template)
    prediction, intermediates = model.apply({'params': params}, x, times, labels,
                                            mutable=['parity'])
    errors = {}

    def compare(name, actual, expected, atol=2e-4, rtol=2e-4):
        actual, expected = np.asarray(actual), np.asarray(expected)
        errors[name] = float(np.max(np.abs(actual-expected)))
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)

    for name, values in intermediates['parity'].items():
        expected = reference[f'activation/{name}']
        if expected.ndim == 4:
            expected = expected.transpose(0, 2, 3, 1)
        compare(f'layer/{name}', values[0], expected)

    def objective(weights, inputs):
        predicted = model.apply({'params': weights}, inputs, times, labels)
        return jnp.mean((predicted-target)**2)

    loss, (grads, dx) = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(params, x)
    compare('loss', loss, reference['loss'])
    compare('input_gradient', dx, nhwc(reference['input_gradient']))
    expected_grads = convert_parameters(select('gradient/'), template)
    from flax.traverse_util import flatten_dict
    for path, actual in flatten_dict(grads).items():
        compare('gradient/'+'/'.join(path), actual, flatten_dict(expected_grads)[path])
    optimizer = optax.adam(2e-4)
    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    updated = optax.apply_updates(params, updates)
    expected_updated = convert_parameters(select('adam/'), template)
    for path, actual in flatten_dict(updated).items():
        compare('adam/'+'/'.join(path), actual, flatten_dict(expected_updated)[path], atol=1e-5)

    variables = {'params': {'model': params}}
    compare('schedule/betas', diffusion.betas, reference['betas'], atol=1e-8)
    compare('schedule/alpha', jnp.cumprod(1-diffusion.betas), reference['alpha'], atol=1e-6)
    perturbed = diffusion.apply(variables, x, times, target, method=diffusion.perturb_x)
    compare('diffusion/perturb', perturbed, nhwc(reference['perturbed']))
    removed = diffusion.apply(variables, x, times, labels, use_ema=False,
                              method=diffusion.remove_noise)
    compare('diffusion/remove', removed, nhwc(reference['removed']))

    if 'sample_initial' in reference:
        # Shared initial noise and per-step noise remove cross-framework RNG differences.
        initial = nhwc(reference['sample_initial'])
        noises = jnp.asarray(reference['sample_noises'].transpose(0, 1, 3, 4, 2))
        betas = jnp.asarray(reference['betas'])

        @jax.jit
        def sample():
            def step(images, t):
                images = diffusion.apply(variables, images,
                                         jnp.full((len(images),), t), labels,
                                         use_ema=False, method=diffusion.remove_noise)
                return images+jnp.where(t > 0, jnp.sqrt(betas[t]), 0)*noises[t], None
            return jax.lax.scan(step, initial, jnp.arange(len(noises)-1, -1, -1))[0]
        samples = np.asarray(sample())
        # Long trajectories accumulate float32 reduction differences.
        compare('sample', samples, nhwc(reference['sample_output']), atol=2e-2, rtol=2e-2)
        grid = samples[..., 0].transpose(1, 0, 2).reshape(28, -1)
        Image.fromarray(np.uint8(np.clip((grid+1)*127.5, 0, 255))).resize((896, 224)).save(out/'transferred_ddpm.png')

    state = train_state.TrainState.create(apply_fn=diffusion.apply,
                                         params={'model': params}, tx=optimizer)
    with (out/'converted_torch.msgpack').open('wb') as handle:
        pickle.dump(flax.serialization.to_state_dict(state), handle)
    result = dict(passed=True, comparisons=len(errors), max_errors=errors,
                  note='eval/dropout-disabled parity; RNG sequences are shared fixtures')
    (out/'parity.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, comparisons=len(errors),
                         largest_error=max(errors.values()))), flush=True)


if __name__ == '__main__':
    main()
