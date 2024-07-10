import os
import jax
import flax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
from flax import linen as nn
from flax.training import train_state
from flax.training.common_utils import shard
from flax.training import checkpoints
from flax.serialization import to_bytes, from_bytes
import optax
import pickle

from flax.core import FrozenDict
from jax.tree_util import tree_leaves
from flax_ddpm import script_utils
from datasets.flax_tiny_mnist import MnistDataset
from flax_ddpm.script_utils import get_args
from flax_ddpm.script_utils import get_diffusion_from_args
from tools import file_utils
import struct
import numpy as np
from torch.utils.data import DataLoader
import logging
import sys
import imageio

logger = logging.getLogger(__name__)

@jax.jit
def train_step(state, batch, rng):
    x,y = batch
    transformed= x[0].reshape(28, 28, 1)

    x = jnp.array(x).transpose(0, 2, 3, 1)
    y = jnp.array(y)

    def loss_fn(params):
        loss = state.apply_fn({'params': params}, rng, x, y)
        return loss
    gradient_fn = jax.value_and_grad(loss_fn)
    loss, grads = gradient_fn(state.params) 
    state = state.apply_gradients(grads=grads) 
    return  loss, state

@jax.jit
def eval_step(state, batch, rng):
    x,y = batch
    x = jnp.array(x).transpose(0, 2, 3, 1)
    y = jnp.array(y)
    loss = state.apply_fn({'params': state.params}, rng, x, y)
    return loss


# prepare data
def numpy_collate(batch):
  if isinstance(batch[0], jnp.ndarray):
    return np.stack(batch)
  elif isinstance(batch[0], (tuple,list)):
    transposed = zip(*batch)
    return [numpy_collate(samples) for samples in transposed]
  else:
    if isinstance(batch, (tuple,list)):
        if not isinstance(batch[0], int):
            batch=[jnp.array(im).transpose(1,2,0) for im in batch]
    return jnp.array(batch)


def main(args):
    file_utils.mkdir(args.log_dir)
    try: 
        rng = jax.random.PRNGKey(0)
        rng, x_rng = jax.random.split(rng)

        diffusion = get_diffusion_from_args(args)
        batch_size = args.batch_size

        #load dataset
        target_labels = list(range(args.num_classes))
        train_dataset = MnistDataset(is_train=True, target_labels=target_labels)
        test_dataset = MnistDataset(is_train=False, target_labels=target_labels)
        train_loader = script_utils.cycle(DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=numpy_collate,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        ))
        test_loader = DataLoader(test_dataset, batch_size=batch_size,collate_fn=numpy_collate, shuffle=True, num_workers=0)

        #init
        x = jax.random.normal(x_rng, (128,28,28,1))
        y = jnp.ones(batch_size, dtype=jnp.int32)
        rng, state_rng = jax.random.split(rng)
        variables = diffusion.init(rngs=state_rng,rng=rng, x=x,y=y)# Initialize the parameters
        optimizer = optax.adam(learning_rate=args.learning_rate)


        state = train_state.TrainState.create(
                apply_fn=diffusion.apply,
                params=variables['params'],
                tx=optimizer,
            )
        print("Run start.")
        for iteration in range(1, args.iterations + 1):
            #train
            batch = next(train_loader)
            rng, batch_rng = jax.random.split(rng)
            loss, state = train_step(state, batch, rng)
            print(f"=====> iter: {iteration}, loss: {round(loss, 6)}")
            
            #eval
            if iteration % args.log_rate == 0:
                test_loss = 0
                for batch in test_loader:
                    batch_rng, rngs = jax.random.split(rng)
                    test_loss += eval_step(state, batch, batch_rng)
                test_loss /= len(test_loader)
                print(f"---------> test loss: {round(test_loss, 6)}")

            if iteration % args.checkpoint_rate == 0:
                model_filename = os.path.join(args.log_dir, f"{args.project_name}-{args.run_name}-iteration-{iteration}-model.msgpack")
                state_dict=flax.serialization.to_state_dict(state)
                pickle.dump(state_dict,open(model_filename,"wb"))
              
        print("Run finished.")

    except KeyboardInterrupt:
        print("Keyboard interrupt, run finished early")


if __name__ == "__main__":
    args = get_args()
    for k, v in args.__dict__.items():
        print(f"===> {k} : {v}")
    main(args)
