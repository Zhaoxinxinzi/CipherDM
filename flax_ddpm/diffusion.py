import flax
from flax import linen as nn
from functools import partial
from copy import deepcopy
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from flax_ddpm.unet import UNet
import sys
def extract(a, t, x_shape):
    """
    :param a:
    :param t:
    :param x_shape:
    :return:
    """
    b, *_ = t.shape
    out = jnp.take(a, t, axis=-1)
    # return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    return jnp.reshape(out, (b,) + (1,) * (len(x_shape) - 1))


class EMA():
    def __init__(self, decay):
        self.decay = decay
    
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_params, current_params):
        return jax.tree_multimap(self.update_average, ema_params, current_params)



class FlaxGaussianDiffusion(nn.Module):
    model: UNet
    img_size: tuple
    img_channels: int
    num_classes: int
    betas: jnp.ndarray
    loss_type: str = "l2"
    ema_decay: float = 0.9999
    ema_start: int = 5000
    ema_update_rate: int = 1

    def setup(self):
        self.ema_model = deepcopy(self.model)

        self.ema = EMA(self.ema_decay)
        self.step = 0

        if self.loss_type not in ["l1", "l2"]:
            raise ValueError("setup() got unknown loss type")

        self.num_timesteps = len(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)

        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1 - self.alphas_cumprod)
        self.reciprocal_sqrt_alphas = jnp.sqrt(1 / self.alphas)

        self.remove_noise_coeff = self.betas / jnp.sqrt(1 - self.alphas_cumprod)
        self.sigma = jnp.sqrt(self.betas)


    def update_ema(self, ema_params, current_params, step):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                ema_params = current_params
            else:
                ema_params = self.ema.update_model_average(ema_params, current_params)
        return ema_params, step

    def remove_noise(self, x, t, y, use_ema=False):
        remove_noise_coeff = extract(self.remove_noise_coeff, t, x.shape)
        reciprocal_sqrt_alphas = extract(self.reciprocal_sqrt_alphas, t, x.shape)

        if use_ema:
            pred_noise = self.ema_model(x, t, y)
        else:
            pred_noise = self.model(x, t, y)
        # print("pred_noise",pred_noise.transpose(0,2,3,1))

        return (x - remove_noise_coeff * pred_noise) * reciprocal_sqrt_alphas

    def sample(self, batch_size, y=None, use_ema=False, rng_key=None):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")
        # 初始化随机噪声
        rng, x_rng = jax.random.split(rng_key)
        x = jax.random.normal(x_rng, (batch_size, *self.img_size, self.img_channels))
        for t in reversed(range(self.num_timesteps)):
            rng, step_rng = jax.random.split(rng)
            t_batch = jnp.full((batch_size,), t, dtype=jnp.int32)
            x = self.remove_noise(x, t_batch, y, use_ema)
            if t > 0:
                sigma_t = extract(self.sigma, t_batch, x.shape)
                x += sigma_t * jax.random.normal(rng, x.shape)
        # print("xsample",x.transpose(0,3,1,2))
        return x

    def sample_sequence(self, batch_size, y=None, use_ema=False, rng_key=None):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")
        rng, x_rng = jax.random.split(rng_key)
        x = jax.random.normal(x_rng, (batch_size, *self.img_size, self.img_channels))
        for t in reversed(range(self.num_timesteps)):
            rng, step_rng = jax.random.split(rng)
            # step_rng = jnp.asarray(step_rng)
            t_batch = jnp.full((batch_size,), t, dtype=jnp.int32)
            x = self.remove_noise(x, t_batch, y, use_ema)
            if t > 0:
                sigma_t = extract(self.sigma, t_batch, x.shape)
                noise = jax.random.normal(step_rng, x.shape)
                # print("noise",noise.transpose(0,3,1,2))
                x += sigma_t * noise
            # print("yes",x.transpose(0,3,1,2))
            yield x

    def sample_ddim(self, batch_size, y=None, use_ema= False, rng_key=None, ddim_timesteps =50, ddim_discr_method = "uniform", clip_denoised=True):
        ddim_eta=0.0 #eta=1, ddpm; eta=0, ddim
        if ddim_discr_method == 'uniform':
            c = self.num_timesteps // ddim_timesteps
            ddim_timestep_seq = jnp.asarray(jnp.arange(0, self.num_timesteps, c))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                jnp.square(jnp.linspace(0, jnp.sqrt(self.num_timesteps * .8), ddim_timesteps))
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = jnp.append(jnp.array([0]), ddim_timestep_seq[:-1])
        # start from pure noise (for each example in the batch)
        rng, x_rng = jax.random.split(rng_key)
        x = jax.random.normal(x_rng, (batch_size, *self.img_size, self.img_channels))
        for t in reversed(range(0, ddim_timesteps)):
            rng, step_rng = jax.random.split(rng)
            t_batch = jnp.full((batch_size,), ddim_timestep_seq[t], dtype=jnp.int32)
            prev_t_batch = jnp.full((batch_size,), ddim_timestep_prev_seq[t], dtype=jnp.int32)
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = extract(self.alphas_cumprod, t_batch, x.shape)
            alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t_batch, x.shape)
            # 2. predict noise using model
            # pred_noise = self.model(x, t_batch, y)
            if use_ema:
                pred_noise = self.ema_model(x, t_batch, y)
            else:
                pred_noise = self.model(x, t_batch, y)          
            # 3. get the predicted x_0
            pred_x0 = (x - jnp.sqrt((1. - alpha_cumprod_t)) * pred_noise) / jnp.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = jnp.clip(pred_x0, a_min=-1., a_max=1.)
            # 4. compute variance: "sigma_t(η)"
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * jnp.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))           
            # 5. compute "direction pointing to x_t" 
            pred_dir_xt = jnp.sqrt(1 - alpha_cumprod_t_prev - jnp.square(sigmas_t)) * pred_noise         
            # 6. compute x_{t-1}
            x_prev = jnp.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * jax.random.normal(rng, x.shape)
            x = x_prev
        return x

    def perturb_x(self, x, t, noise): # q_sample
        sqrt_alphas_cumprod = extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise
    

    def get_losses(self, x, t, y, rng_key):
        rng, rng_key = jax.random.split(rng_key)
        noise = jax.random.normal(rng, x.shape)
        perturbed_x = self.perturb_x(x, t, noise)
        estimated_noise = self.model(perturbed_x, t, y)

        if self.loss_type == "l1":
            loss = jnp.mean(jnp.abs(estimated_noise - noise))
        elif self.loss_type == "l2":
            loss = jnp.mean(jnp.square(estimated_noise - noise))
        return loss

        

    def __call__(self,rng, x, y=None):
        b, h, w, c = x.shape
        if h != self.img_size[0] or w != self.img_size[1]:
            raise ValueError("Image dimensions do not match diffusion parameters")
        rng, rng_key = jax.random.split(rng)
        t = random.randint(rng, (b,), 0, self.num_timesteps)
        loss = self.get_losses(x, t, y, rng_key)
        # print("call_loss",loss)
        return loss


def generate_cosine_schedule(T, s=0.008):
    def f(t, T):
        return (jnp.cos((t / T + s) / (1 + s) * jnp.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T)

    for t in range(T + 1):
        alphas.append(f(t, T) / f0)
    
    betas = []

    for t in range(1, T + 1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
    
    return jnp.array(betas)


def generate_linear_schedule(T, low, high):
    return jnp.linspace(low, high, T)
