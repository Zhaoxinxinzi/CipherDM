"""Train from local MNIST IDX data; no torch/torchvision dependency."""
import argparse
import json
import os
from pathlib import Path
import pickle
import time

import flax
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax
from PIL import Image

from flax_ddpm.script_utils import get_args, get_diffusion_from_args


def read_mnist(root, prefix, classes):
    images = np.fromfile(root / f"{prefix}-images-idx3-ubyte", np.uint8, offset=16)
    images = images.reshape(-1, 28, 28, 1)
    labels = np.fromfile(root / f"{prefix}-labels-idx1-ubyte", np.uint8, offset=8)
    mask = labels < classes
    return images[mask].astype(np.float32) / 127.5 - 1, labels[mask].astype(np.int32)


def atomic_pickle(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist_raw", required=True)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--preview_sampler", choices=['ddpm', 'ddim'], default='ddpm')
    args = get_args(parser)
    if args.model_checkpoint or args.optim_checkpoint:
        raise ValueError("This entry point starts fresh; resume is not implemented")
    if args.norm == 'bn':
        raise ValueError("This TrainState entry point supports gn/in; bn needs batch_stats state")
    root, out = Path(args.mnist_raw), Path(args.log_dir)
    out.mkdir(parents=True, exist_ok=True)
    x_train, y_train = read_mnist(root, "train", args.num_classes)
    x_test, y_test = read_mnist(root, "t10k", args.num_classes)
    if len(x_train) < args.batch_size or len(x_test) < 16:
        raise ValueError("Not enough matching training/test samples")
    config = dict(vars(args), devices=[str(d) for d in jax.devices()],
                  jax_version=jax.__version__, flax_version=flax.__version__,
                  train_samples=len(x_train), test_samples=len(x_test),
                  effective_num_res_blocks=args.num_res_blocks,
                  architecture="torch-aligned-v1",
                  upsampling="nearest resize + Conv2d", ema_start=2000,
                  pid=os.getpid())
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2), flush=True)
    diffusion = get_diffusion_from_args(args)
    key = jax.random.PRNGKey(args.seed)
    key, init_key, loss_key = jax.random.split(key, 3)
    params = diffusion.init(init_key, loss_key,
                            jnp.asarray(x_train[:args.batch_size]),
                            jnp.asarray(y_train[:args.batch_size]))["params"]
    state = train_state.TrainState.create(apply_fn=diffusion.apply, params=params,
                                         tx=optax.adam(args.learning_rate))
    ema = state.params

    @jax.jit
    def train_step(current, average, images, labels, rng):
        def objective(weights):
            loss_rng, dropout_rng = jax.random.split(rng)
            return diffusion.apply({'params': weights}, loss_rng, images, labels,
                                   deterministic=False, rngs={'dropout': dropout_rng})
        loss, grads = jax.value_and_grad(objective)(current.params)
        current = current.apply_gradients(grads=grads)
        decay = jnp.where(current.step < 2000, 0., args.ema_decay)
        candidate = jax.tree_util.tree_map(
            lambda old, new: decay * old + (1-decay) * new, average, current.params)
        average = jax.tree_util.tree_map(
            lambda old, new: jnp.where(current.step % args.ema_update_rate == 0, new, old),
            average, candidate)
        return current, average, loss

    @jax.jit
    def predict(weights, images, times, labels):
        return diffusion.apply({"params": weights}, images, times, labels,
                               method=lambda m, x, t, y: m.model(x, t, y))

    alpha = jnp.cumprod(1 - diffusion.betas)
    test_x, test_y = jnp.asarray(x_test[:16]), jnp.asarray(y_test[:16])
    test_noise = jax.random.normal(jax.random.PRNGKey(12), test_x.shape)

    def evaluate(weights):
        result = {}
        for t in [0, 10, 100, 500, 999]:
            if t >= args.num_timesteps:
                continue
            noisy = jnp.sqrt(alpha[t])*test_x + jnp.sqrt(1-alpha[t])*test_noise
            pred = predict(weights, noisy, jnp.full((16,), t), test_y)
            result[str(t)] = float(jnp.mean((pred-test_noise)**2))
        return result

    # Compile one denoiser in a scan. DDPM matches the Torch reference sampler.
    @jax.jit
    def preview(weights):
        labels = jnp.arange(8, dtype=jnp.int32) % args.num_classes
        images = jax.random.normal(jax.random.PRNGKey(0), (8, 28, 28, 1))
        count = args.num_timesteps if args.preview_sampler == 'ddpm' else min(50, args.num_timesteps)
        times = jnp.asarray(np.linspace(0, args.num_timesteps-1, count, dtype=np.int32))
        previous = jnp.concatenate([jnp.array([-1]), times[:-1]])

        def step(carry, pair):
            images, rng = carry
            t, prev = pair
            noise = predict(weights, images, jnp.full((8,), t), labels)
            rng, step_rng = jax.random.split(rng)
            if args.preview_sampler == 'ddpm':
                beta = diffusion.betas[t]
                images = (images-beta/jnp.sqrt(1-alpha[t])*noise)/jnp.sqrt(1-beta)
                images += jnp.where(t > 0, jnp.sqrt(beta), 0)*jax.random.normal(step_rng, images.shape)
            else:
                clean = jnp.clip((images-jnp.sqrt(1-alpha[t])*noise)/jnp.sqrt(alpha[t]), -1, 1)
                prev_alpha = jnp.where(prev < 0, 1., alpha[jnp.maximum(prev, 0)])
                images = jnp.sqrt(prev_alpha)*clean+jnp.sqrt(1-prev_alpha)*noise
            return (images, rng), None
        (images, _), _ = jax.lax.scan(step, (images, jax.random.PRNGKey(1)),
                                     (times[::-1], previous[::-1]))
        return images

    def save(step):
        snapshot = flax.serialization.to_state_dict(state)
        for name, weights in [("raw", state.params), ("ema", ema)]:
            snapshot["params"] = flax.serialization.to_state_dict(weights)
            atomic_pickle(out / f"step_{step:07d}_{name}.msgpack", snapshot)
        metrics = dict(step=step, raw=evaluate(state.params), ema=evaluate(ema))
        with (out / "denoising.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        samples = np.asarray(preview(ema))[..., 0]
        grid = samples.reshape(2, 4, 28, 28).transpose(0, 2, 1, 3).reshape(56, 112)
        Image.fromarray(np.uint8(np.clip((grid+1)*127.5, 0, 255))).resize((448, 224)).save(
            out / f"preview_{step:07d}.png")
        print("CHECKPOINT " + json.dumps(metrics), flush=True)

    random = np.random.default_rng(args.seed)
    start = time.monotonic()
    accumulated = 0.
    permutation = random.permutation(len(x_train))
    cursor = 0
    print("TRAIN_START", flush=True)
    for iteration in range(1, args.iterations + 1):
        if cursor + args.batch_size > len(x_train):
            permutation = random.permutation(len(x_train))
            cursor = 0
        indices = permutation[cursor:cursor+args.batch_size]
        cursor += args.batch_size
        key, batch_key = jax.random.split(key)
        state, ema, loss = train_step(state, ema, x_train[indices], y_train[indices], batch_key)
        loss = float(loss)
        if not np.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {iteration}")
        accumulated += loss
        if iteration == 1 or (iteration <= 100 and iteration % 10 == 0) or iteration % 100 == 0:
            print(json.dumps(dict(step=iteration, loss=loss,
                                  mean_loss=accumulated / iteration,
                                  elapsed_seconds=round(time.monotonic()-start, 2))), flush=True)
        if iteration == 1 or iteration % args.checkpoint_rate == 0 or iteration == args.iterations:
            save(iteration)
    (out / "completed.json").write_text(json.dumps(dict(step=args.iterations,
        elapsed_seconds=time.monotonic()-start)) + "\n")
    print("TRAIN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
