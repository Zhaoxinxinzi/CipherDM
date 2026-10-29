# CipherDM
The first 3PC framework for Diffusion Model sampling.

## Execute

```console
docker build -t myubuntu:v1 --build-arg http_proxy=## --build-arg https_proxy=## --network host .
docker run --rm --network=host myubuntu:v1
```

# Flax training and diagnostics

Run commands from the repository root. The validated environment uses Python 3.10,
JAX 0.4.18, Flax 0.7.0, Optax 0.1.4, NumPy, and Pillow. Install a GPU-compatible
jaxlib and cuDNN matching that build. The training and audit scripts do not need
Torch, torchvision, or SPU.

Activate the environment and verify that JAX detects the GPU:

```bash
conda activate flax
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -c 'import jax; print(jax.devices())'
```

If an older CUDA installation shadows a compatible system cuDNN, put the directory
containing the compatible library first in `LD_LIBRARY_PATH`. On the machine used
for validation, `/usr/lib/x86_64-linux-gnu` supplies cuDNN 8.9.2; JAX requires at
least 8.6. `JAX_PLATFORMS=cuda` makes initialization fail instead of silently
falling back to CPU.

Set `MNIST_RAW` to a directory containing the four uncompressed standard MNIST IDX
files: `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`,
`t10k-images-idx3-ubyte`, and `t10k-labels-idx1-ubyte`.

```bash
export MNIST_RAW=/path/to/MNIST/raw
python -u scripts/flax_train.py \
  --mnist_raw "$MNIST_RAW" \
  --log_dir checkpoints \
  --iterations 20000 --batch_size 128 \
  --learning_rate 0.0002 --dropout 0.1 --checkpoint_rate 1000
```

This starts from fresh parameters using labels 0 and 1, fixed seed 20261029,
and the Torch-aligned network architecture. Dropout is enabled only for training
and receives an independent RNG. The model factory explicitly defaults to two
residual blocks per level, matching the original Torch factory's effective
configuration. EMA is maintained outside the Flax
module: weights are copied before step 2000, then averaged with decay 0.999.

The output directory contains `config.json`, raw and EMA checkpoints,
`denoising.jsonl` with fixed-noise MSE at selected timesteps, and 1000-step DDPM
preview grids. Use `--preview_sampler ddim` for 50-step DDIM previews. Preview
labels alternate 0 and 1. Checkpoints and previews are
written at step 1, each checkpoint interval, and the final step.
`completed.json` marks successful completion. Despite the `.msgpack` extension,
checkpoints retain the existing pickled Flax TrainState dictionary format.
An EMA checkpoint contains EMA parameters with the raw optimizer state and is
intended for inference. This entry point does not implement training resumption.

Audit a checkpoint:

```bash
python scripts/audit_flax.py \
  --mnist_raw "$MNIST_RAW" \
  --model_path checkpoints/step_0020000_ema.msgpack
```

Add `--audit_train_steps 20` for an in-memory optimization probe. This uses a
small test batch to check gradient updates and does not overwrite the checkpoint;
its loss change is not a generalization measurement.

Sample using the ordinary JAX entry point:

```bash
python scripts/flax_sample.py \
  --model_path checkpoints/step_0020000_ema.msgpack \
  --save_dir model_eval
```

This saves label-1 images and a grid to `model_eval/` using DDPM. Use `--label`,
`--num_samples`, and `--sampler ddim` to customize sampling. Both separate EMA
checkpoints and checkpoints containing an `ema_params` field are supported.
Supply the same architecture and diffusion arguments when training with
nondefault settings. Legacy ConvTranspose checkpoints are explicitly rejected.
The ordinary sampler uses Pillow and does not require OpenCV or torchvision.
SPU evaluation remains a
separate entry point with its existing configuration requirements.

See [FLAX_AUDIT.md](FLAX_AUDIT.md) for the original checkpoint investigation and
the distinction between confirmed defects and unverified causes.
See [TORCH_PARITY.md](TORCH_PARITY.md) for weight conversion and numerical
comparisons with the original Torch source.
