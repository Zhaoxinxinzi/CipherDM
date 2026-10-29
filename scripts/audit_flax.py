"""Check an existing checkpoint against local MNIST IDX files, without torchvision.

Run from the repo root with PYTHONPATH=.; accepts the usual model arguments.
"""
import argparse
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax_ddpm.script_utils import get_args, get_diffusion_from_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--mnist_raw", required=True)
    parser.add_argument("--audit_batch_size", type=int, default=16)
    parser.add_argument("--audit_train_steps", type=int, default=0)
    parser.add_argument("--audit_output", default=None)
    args = get_args(parser)
    root = Path(args.mnist_raw)
    images = np.fromfile(root / "t10k-images-idx3-ubyte", dtype=np.uint8, offset=16)
    images = images.reshape(-1, 28, 28, 1)
    labels = np.fromfile(root / "t10k-labels-idx1-ubyte", dtype=np.uint8, offset=8)
    indices = np.flatnonzero(labels < args.num_classes)[:args.audit_batch_size]
    if len(indices) == 0:
        raise ValueError("No matching MNIST labels")
    x = jnp.asarray(images[indices] / 127.5 - 1, dtype=jnp.float32)
    y = jnp.asarray(labels[indices], dtype=jnp.int32)
    # Existing training checkpoints are pickle state_dicts despite .msgpack suffix.
    with open(args.model_path, "rb") as handle:
        checkpoint = pickle.load(handle)
    params = jax.tree_util.tree_map(jnp.asarray, checkpoint["params"])
    diffusion = get_diffusion_from_args(args)

    @jax.jit
    def predict(weights, image, timestep):
        return diffusion.apply(
            {"params": weights}, image, timestep, y,
            method=lambda module, image, timestep, labels:
                module.model(image, timestep, labels),
        )

    noise = jax.random.normal(jax.random.PRNGKey(12), x.shape)
    alpha = jnp.cumprod(1 - diffusion.betas)
    metrics = []
    for t in sorted(set(min(t, args.num_timesteps - 1) for t in [0, 10, 100, 500, 999])):
        xt = jnp.sqrt(alpha[t]) * x + jnp.sqrt(1 - alpha[t]) * noise
        prediction = predict(params, xt, jnp.full((len(x),), t, dtype=jnp.int32))
        metrics.append(dict(t=t, noise_mse=float(jnp.mean((prediction-noise)**2)),
                            zero_baseline_mse=float(jnp.mean(noise**2)),
                            prediction_mean=float(prediction.mean()),
                            prediction_std=float(prediction.std())))
    result = dict(checkpoint=str(args.model_path), step=int(checkpoint["step"]),
                  parameter_count=sum(a.size for a in jax.tree_util.tree_leaves(params)),
                  devices=[str(device) for device in jax.devices()], metrics=metrics)

    if args.audit_train_steps:
        # In-memory probe only: never overwrite the checkpoint. This is a small
        # optimization check on a fixed test batch, not a generalization score.
        optimizer = optax.adam(args.learning_rate)
        optimizer_state = optimizer.init(params)
        initial_params = params
        key = jax.random.PRNGKey(42)

        @jax.jit
        def loss_fn(weights, rng):
            return diffusion.apply({"params": weights}, rng, x, y)

        @jax.jit
        def update(weights, opt_state, rng):
            loss, grads = jax.value_and_grad(loss_fn)(weights, rng)
            updates, opt_state = optimizer.update(grads, opt_state, weights)
            return optax.apply_updates(weights, updates), opt_state, loss

        before = float(loss_fn(params, key))
        for _ in range(args.audit_train_steps):
            key, step_key = jax.random.split(key)
            params, optimizer_state, _ = update(params, optimizer_state, step_key)
        deltas = jax.tree_util.tree_map(lambda a, b: jnp.sum((a-b)**2), params, initial_params)
        result["optimization_probe"] = dict(
            steps=args.audit_train_steps,
            fixed_key_loss_before=before,
            fixed_key_loss_after=float(loss_fn(params, jax.random.PRNGKey(42))),
            parameter_delta_l2=float(jnp.sqrt(sum(jax.tree_util.tree_leaves(deltas)))),
        )
    output = json.dumps(result, indent=2)
    print(output, flush=True)
    if args.audit_output:
        Path(args.audit_output).write_text(output + "\n")


if __name__ == "__main__":
    main()
