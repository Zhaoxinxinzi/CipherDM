"""Export a Torch reference fixture without requiring Torch in the Flax env."""
import argparse
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--torch_root', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--sample_steps', type=int, default=1000)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.torch_root).resolve()))
    import torch
    from ddpm.script_utils import get_args, get_diffusion_from_args
    torch.set_num_threads(4)
    torch.manual_seed(7)
    sys.argv = [sys.argv[0]]
    config = get_args()
    diffusion = get_diffusion_from_args(config).cpu().eval()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        diffusion.load_state_dict(state)
        diffusion.model.load_state_dict(diffusion.ema_model.state_dict())
    model = diffusion.model

    def array(tensor):
        # Also works with old PyTorch builds and NumPy 2.x on the export host.
        return np.asarray(tensor.detach().cpu().tolist(), dtype=np.float32)

    fixture = {f'weight/{name}': array(value) for name, value in model.named_parameters()}
    rng = np.random.default_rng(17)
    x = rng.normal(size=(4, 1, 28, 28)).astype(np.float32)
    target = rng.normal(size=x.shape).astype(np.float32)
    times, labels = np.array([0, 10, 500, 999], np.int64), np.array([0, 1, 0, 1], np.int64)
    fixture.update(x=x, target=target, times=times, labels=labels,
                   betas=array(diffusion.betas), alpha=array(diffusion.alphas_cumprod))

    def hook(name):
        def save(module, inputs, output):
            fixture[f'activation/{name}'] = array(output)
        return save

    handles = [model.time_mlp.register_forward_hook(hook('time_embedding')),
               model.init_conv.register_forward_hook(hook('input_conv')),
               model.out_conv.register_forward_hook(hook('output'))]
    for name, modules in [('down', model.downs), ('middle', model.mid), ('up', model.ups)]:
        for index, module in enumerate(modules):
            handles.append(module.register_forward_hook(hook(f'{name}_{index}')))
    tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    prediction = model(tensor, torch.tensor(times, dtype=torch.long), torch.tensor(labels, dtype=torch.long))
    loss = ((prediction-torch.tensor(target, dtype=torch.float32))**2).mean()
    loss.backward()
    fixture['loss'] = array(loss)
    fixture['input_gradient'] = array(tensor.grad)
    for name, value in model.named_parameters():
        fixture[f'gradient/{name}'] = array(value.grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    optimizer.step()
    for name, value in model.named_parameters():
        fixture[f'adam/{name}'] = array(value)
    for handle in handles:
        handle.remove()
    # Restore weights before the sampling comparison.
    model.load_state_dict({name: torch.tensor(fixture[f'weight/{name}'], dtype=torch.float32)
                           for name, _ in model.named_parameters()})
    with torch.no_grad():
        tx = torch.tensor(x, dtype=torch.float32)
        tt, ty = torch.tensor(times, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
        fixture['perturbed'] = array(diffusion.perturb_x(tx, tt, torch.tensor(target, dtype=torch.float32)))
        fixture['removed'] = array(diffusion.remove_noise(tx, tt, ty, use_ema=False))
    if args.sample_steps:
        initial = rng.normal(size=(4, 1, 28, 28)).astype(np.float32)
        noises = rng.normal(size=(args.sample_steps, 4, 1, 28, 28)).astype(np.float32)
        sample = torch.tensor(initial, dtype=torch.float32)
        with torch.no_grad():
            for t in reversed(range(args.sample_steps)):
                sample = diffusion.remove_noise(sample, torch.full((4,), t),
                                                torch.tensor(labels, dtype=torch.long), use_ema=False)
                if t > 0:
                    sample += diffusion.sigma[t] * torch.tensor(noises[t], dtype=torch.float32)
        fixture.update(sample_initial=initial, sample_noises=noises,
                       sample_output=array(sample))
    np.savez_compressed(args.output, **fixture)
    print(f'Exported {len(fixture)} reference arrays to {args.output}', flush=True)


if __name__ == '__main__':
    main()
