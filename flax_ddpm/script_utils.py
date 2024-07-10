import jax
from flax import linen as nn
import flax
import argparse
import datetime
from flax_ddpm.unet import UNet
from flax_ddpm.diffusion import (
    FlaxGaussianDiffusion,
    generate_linear_schedule,
    generate_cosine_schedule,
)


def cycle(dl):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for data in dl:
            yield data

def diffusion_defaults():
    defaults = dict(
        num_timesteps=1000,#1000
        schedule="linear",
        loss_type="l2",
        use_labels=False,

        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        time_emb_dim=128 * 4,
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(1,),

        ema_decay=0.9999,
        ema_update_rate=1,
    )

    return defaults


def get_diffusion_from_args(args):
    # Define Flax-compatible activation functions
    activations = {
        "relu": nn.relu,
        "silu": nn.silu,
    }

    model = UNet(
        img_channels=args.img_channels,
        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
        time_emb_dim=args.time_emb_dim,
        norm=args.norm,
        dropout=args.dropout,
        activation=activations[args.activation],
        attention_resolutions=args.attention_resolutions,
        num_classes=args.num_classes,
        initial_pad=0,
        num_groups=args.num_groups
    )

    if args.schedule == "cosine":
        betas = generate_cosine_schedule(args.num_timesteps)
    else:
        betas = generate_linear_schedule(
            args.num_timesteps,
            args.schedule_low * 1000 / args.num_timesteps,
            args.schedule_high * 1000 / args.num_timesteps,
        )

    diffusion = FlaxGaussianDiffusion(
        model,
        args.img_size,
        args.img_channels,
        args.num_classes,
        betas,
        ema_decay=args.ema_decay,
        ema_update_rate=args.ema_update_rate,
        ema_start=2000,
        loss_type=args.loss_type,
    )

    return diffusion

def get_args(parser: argparse.ArgumentParser = None) -> argparse.Namespace:
    """
    Get args for all (u-net and diffusion training super-params.)
    Default for training tiny model with mnist dataset. You can also pass args to override.
    """
    time_frame = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    run_name = f"tiny_mnist_{time_frame}"
    defaults = dict(
        img_channels=1,
        img_size=(28, 28),
        num_classes=2,
        num_groups=2,
        learning_rate=2e-4,
        batch_size=128,
        iterations=20000, #50000
        log_to_wandb=False,
        log_rate=500, #500
        checkpoint_rate=5000, #1000
        log_dir="./checkpoints",
        project_name="aigc-ddpm",
        run_name=run_name,
        model_checkpoint=None,
        optim_checkpoint=None,
        schedule_low=1e-4,
        schedule_high=0.02,
        device="gpu" if jax.devices()[0].platform == 'gpu' else "cpu",
        num_timesteps=1000, #1000
        schedule="linear",
        loss_type="l2",
        base_channels=4,
        channel_mults=(1, 2),
        num_res_blocks=1,
        time_emb_dim=8,
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(1,),
        ema_decay=0.999,
        ema_update_rate=1,
    )

    parser = argparse.ArgumentParser() if parser is None else parser
    add_dict_to_argparser(parser, defaults)  # args could be override by user
    return parser.parse_args()

def add_dict_to_argparser(parser, defaults_dict):
    for key, default_value in defaults_dict.items():
        parser.add_argument(f"--{key}", default=default_value, type=type(default_value))

# Example usage
if __name__ == '__main__':
    args = get_args()
