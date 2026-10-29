import math
import flax
from flax import linen as nn
from functools import partial
from copy import deepcopy
import jax
import jax.numpy as jnp
from jax import random
import numpy as np


def torch_uniform(fan_in):
    """PyTorch Conv2d/Linear reset_parameters distribution (not equal RNG draws)."""
    bound = fan_in ** -0.5
    return lambda key, shape, dtype=jnp.float32: random.uniform(
        key, shape, dtype, minval=-bound, maxval=bound)


def conv(in_channels, out_channels, size=3, strides=(1, 1), padding='SAME'):
    return nn.Conv(out_channels, (size, size), strides=strides, padding=padding,
                   kernel_init=torch_uniform(in_channels * size * size),
                   bias_init=torch_uniform(in_channels * size * size))


def dense(in_features, out_features):
    return nn.Dense(out_features, kernel_init=torch_uniform(in_features),
                    bias_init=torch_uniform(in_features))


def get_norm(norm, num_groups=None, deterministic=True):
    if norm == "in":
        return nn.GroupNorm(num_groups=None, group_size=1, epsilon=1e-5)
    elif norm == "bn":
        return nn.BatchNorm(use_running_average=deterministic, momentum=0.9, epsilon=1e-5)
    elif norm == "gn":
        if num_groups is None:
            raise ValueError("num_groups must be specified for group normalization")
        return nn.GroupNorm(num_groups=num_groups, epsilon=1e-5)
    elif norm is None:
        return lambda x: x  # No normalization
    else:
        raise ValueError("unknown normalization type")
    


class PositionalEmbedding(nn.Module):
    dim: int
    scale: float

    @nn.compact
    def __call__(self, x):
        half_dim = self.dim // 2
        assert self.dim % 2 == 0, "Embedding dimension must be even"

        # Create the embedding factors
        emb = math.log(10000) / half_dim
        emb = jnp.exp(jnp.arange(half_dim) * -emb)

        # Scale and compute the sine and cosine embeddings
        emb = jnp.outer(x * self.scale, emb)
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)

        return emb



class Downsample(nn.Module):
    in_channels: int

    @nn.compact
    def __call__(self, x, time_emb=None, y=None, deterministic=True):
        if x.shape[1] % 2 == 1:
            raise ValueError("Downsampling tensor height should be even")
        if x.shape[2] % 2 == 1:
            raise ValueError("Downsampling tensor width should be even")

        # Define the downsampling convolutional layer
        downsample = conv(self.in_channels, self.in_channels, strides=(2, 2),
                          padding=((1, 1), (1, 1)))

        return downsample(x)



class Upsample(nn.Module):
    in_channels: int

    @nn.compact
    def __call__(self, x, time_emb=None, y=None, deterministic=True):
        # Match torch.nn.Upsample(scale_factor=2, mode='nearest') + Conv2d.
        x = jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)
        return conv(self.in_channels, self.in_channels)(x)



class AttentionBlock(nn.Module):
    in_channels: int
    norm_type: str
    num_groups: int = 32

    def setup(self):
        self.norm = get_norm(self.norm_type, self.num_groups)
        self.to_qkv = conv(self.in_channels, self.in_channels * 3, size=1)
        self.to_out = conv(self.in_channels, self.in_channels, size=1)

    def __call__(self, x, deterministic=True):
        b, h, w, c = x.shape
        normalized = self.norm(x, use_running_average=deterministic) if self.norm_type == 'bn' else self.norm(x)
        qkv = self.to_qkv(normalized)
        q, k, v = jnp.split(qkv, 3, axis=-1)


        q = q.reshape(b, h * w, c)
        k = k.reshape(b, h * w, c).transpose(0, 2, 1)
        v = v.reshape(b, h * w, c)


        dot_products = jnp.matmul(q, k) * (c ** -0.5)
        attention = nn.softmax(dot_products, axis=-1)
        out = jnp.matmul(attention, v)
        out = out.reshape(b, h, w, c)
        return self.to_out(out) + x


class ResidualBlock(nn.Module):
    in_channels: int
    out_channels: int
    dropout: float
    time_emb_dim: int = None
    num_classes: int = None
    norm: str = "gn"
    num_groups: int = 32
    use_attention: bool = False
    activation: callable = nn.relu

    @nn.compact
    def __call__(self, x, time_emb=None, y=None,deterministic=True):
        activation_fn = self.activation  # Replace with your choice of activation

        # First set of layers
        norm_1 = get_norm(self.norm, self.num_groups, deterministic)
        conv_1 = conv(self.in_channels, self.out_channels)
        
        out = activation_fn(norm_1(x))
        out = conv_1(out)

        # Time and class conditioning
        if self.time_emb_dim is not None:
            if time_emb is None:
                raise ValueError("Time conditioning was specified but time_emb is not passed")
            time_bias = dense(self.time_emb_dim, self.out_channels)
            out += time_bias(activation_fn(time_emb))[:, None, None, :]

        if self.num_classes is not None:
            if y is None:
                raise ValueError("Class conditioning was specified but y is not passed")
            class_bias = nn.Embed(self.num_classes, self.out_channels,
                                  embedding_init=nn.initializers.normal(stddev=1.0))
            out += class_bias(y)[:, None, None, :]

        # Second set of layers
        norm_2 = get_norm(self.norm, self.num_groups, deterministic)
        conv_2 = conv(self.out_channels, self.out_channels)

        out = activation_fn(norm_2(out))
        out = nn.Dropout(rate=self.dropout,deterministic=deterministic)(out)
        out = conv_2(out)

        # Residual connection
        residual_connection = conv(self.in_channels, self.out_channels, size=1) if self.in_channels != self.out_channels else lambda x: x
        out += residual_connection(x)

        # Attention block
        if self.use_attention:
            out = AttentionBlock(self.out_channels, self.norm, self.num_groups)(out, deterministic)

        return out




class UNet(nn.Module):
    img_channels: int
    base_channels: int
    channel_mults: tuple = (1, 2, 4, 8)
    num_res_blocks: int = 2
    time_emb_dim: int = None
    time_emb_scale: float = 1.0
    num_classes: int = None
    dropout: float = 0.1
    attention_resolutions: tuple = ()
    norm: str = "gn"
    num_groups: int = 32
    initial_pad: int = 0
    activation: callable = nn.relu
    def setup(self):
        self.time_mlp = None
        if self.time_emb_dim is not None:
            self.time_mlp = nn.Sequential([
                PositionalEmbedding(self.base_channels, self.time_emb_scale),
                dense(self.base_channels, self.time_emb_dim),
                nn.silu,
                dense(self.time_emb_dim, self.time_emb_dim),
            ])
        # print("time_mlp",self.time_mlp)
        self.init_conv = conv(self.img_channels, self.base_channels)

        # Define downsampling and upsampling blocks    
        channels = [self.base_channels]
        now_channels = self.base_channels

        downs = []
        for i, mult in enumerate(self.channel_mults):
            out_channels = self.base_channels * mult

            for _ in range(self.num_res_blocks):
                downs.append(ResidualBlock(
                    now_channels,
                    out_channels,
                    self.dropout,
                    time_emb_dim=self.time_emb_dim,
                    num_classes=self.num_classes,
                    norm=self.norm,
                    num_groups=self.num_groups,
                    use_attention=i in self.attention_resolutions,
                    activation = self.activation
                ))
                now_channels = out_channels
                channels.append(now_channels)

            if i != len(self.channel_mults) - 1:
                downs.append(Downsample(now_channels))
                channels.append(now_channels)
        self.downs = downs

        self.mid = [
            ResidualBlock(
                now_channels,
                now_channels,
                self.dropout,
                time_emb_dim=self.time_emb_dim,
                num_classes=self.num_classes,
                norm=self.norm,
                num_groups=self.num_groups,
                use_attention=True,
                activation = self.activation
            ),
            ResidualBlock(
                now_channels,
                now_channels,
                self.dropout,
                time_emb_dim=self.time_emb_dim,
                num_classes=self.num_classes,
                norm=self.norm,
                num_groups=self.num_groups,
                use_attention=False,
                activation = self.activation
            ),
        ]

        ups=[]
        for i, mult in reversed(list(enumerate(self.channel_mults))):
            out_channels = self.base_channels * mult

            for _ in range(self.num_res_blocks + 1):
                ups.append(ResidualBlock(
                    channels.pop() + now_channels,
                    out_channels,
                    self.dropout,
                    time_emb_dim=self.time_emb_dim,
                    num_classes=self.num_classes,
                    norm=self.norm,
                    num_groups=self.num_groups,
                    use_attention=i in self.attention_resolutions,
                    activation = self.activation
                ))
                now_channels = out_channels

            if i != 0:
                ups.append(Upsample(now_channels))
        self.ups = ups 

        assert len(channels) == 0

        self.out_norm = get_norm(self.norm, self.num_groups)
        self.out_conv = conv(self.base_channels, self.img_channels)

    def __call__(self, x, time=None, y=None, deterministic=True):
        ip = self.initial_pad
        if ip != 0:
            x = jnp.pad(x, ((0, 0), (ip, ip), (ip, ip), (0, 0)))

        if self.time_mlp is not None:
            if time is None:
                raise ValueError("Time conditioning was specified but time is not passed")
            time_emb = self.time_mlp(time)
            self.sow('parity', 'time_embedding', time_emb)
        else:
            time_emb = None
        if self.num_classes is not None and y is None:
            raise ValueError("Class conditioning was specified but y is not passed")
        x = self.init_conv(x)
        self.sow('parity', 'input_conv', x)

        skips = [x]

        for index, layer in enumerate(self.downs):
            x = layer(x, time_emb, y, deterministic=deterministic)
            self.sow('parity', f'down_{index}', x)
            skips.append(x)

        for index, layer in enumerate(self.mid):
            x = layer(x, time_emb, y, deterministic=deterministic)
            self.sow('parity', f'middle_{index}', x)

        for index, layer in enumerate(self.ups):
            if isinstance(layer, ResidualBlock):
                x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = layer(x, time_emb, y, deterministic=deterministic)
            self.sow('parity', f'up_{index}', x)

        normalized = self.out_norm(x, use_running_average=deterministic) if self.norm == 'bn' else self.out_norm(x)
        x = self.activation(normalized)
        x = self.out_conv(x)
        self.sow('parity', 'output', x)

        if self.initial_pad != 0:
            return x[:, ip:-ip, ip:-ip, :]
        else:
            return x
