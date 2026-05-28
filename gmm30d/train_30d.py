"""
Parametric 30D GMM flow-matching trainer.

Differences from `flow_matching_logi.train_model_30d()`:
  - `--steps`, `--ckpt`, `--seed`, `--vis_interval`, `--num_samples`, `--gpu` flags
  - logs throughput every N steps so you can estimate the runtime up-front
  - PCA visualizations saved with the ckpt suffix to avoid name collisions

Examples:
  python train_30d.py --steps 1600000 --ckpt fm_model_30d_2x.pt --seed 42 --gpu 0
  python train_30d.py --steps 4000000 --ckpt fm_model_30d_5x.pt --seed 43 --gpu 1
"""

import os
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow_matching_logi import (
    FlowMatchingOT, GMMDataset30D, set_seed, visualize_samples_pca,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=800_000,
                   help="Number of training optimizer steps")
    p.add_argument("--ckpt", type=str, default="fm_model_30d.pt",
                   help="Checkpoint file name (saved under results30d/)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vis_interval", type=int, default=5_000)
    p.add_argument("--num_samples", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_blocks", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sampling_steps", type=int, default=32)
    p.add_argument("--gpu", type=int, default=None,
                   help="If set, override CUDA_VISIBLE_DEVICES")
    p.add_argument("--save_dir", type=str, default="results30d")
    p.add_argument("--throughput_window", type=int, default=200,
                   help="Window (in steps) over which to average throughput")
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_30d] device={device}, ckpt={args.ckpt}, steps={args.steps}, seed={args.seed}")

    ds = GMMDataset30D(num_samples=args.num_samples, dim=30)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, pin_memory=(device.type == "cuda"),
                        num_workers=0)

    model = FlowMatchingOT(
        dim=30, hidden_dim=args.hidden_dim, num_blocks=args.num_blocks,
        sigma=0.0, lr=args.lr, device=device,
        base_dist="logistic", base_loc=0.0, base_scale=1.0,
    )
    model.to(device)

    os.makedirs(args.save_dir, exist_ok=True)
    suffix = os.path.splitext(args.ckpt)[0].replace("fm_model_30d", "")
    suffix = suffix if suffix else ""              # e.g. "_2x"

    it = iter(loader)
    model.train()

    t_window_start = time.time()
    losses = []
    for step in tqdm(range(1, args.steps + 1), desc=f"train{suffix or '_1x'}"):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = batch.to(device, non_blocking=True)
        loss = model.forward(batch)
        model.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.optimizer.step()
        losses.append(loss.item())

        if step % args.throughput_window == 0:
            elapsed = time.time() - t_window_start
            throughput = args.throughput_window / elapsed if elapsed > 0 else float('nan')
            t_window_start = time.time()
            mean_loss = float(np.mean(losses[-args.throughput_window:]))
            print(f"[step {step}/{args.steps}] loss={mean_loss:.4f} "
                  f"throughput={throughput:.1f} steps/s")

        if step % args.vis_interval == 0:
            model.eval()
            with torch.no_grad():
                idx = np.random.choice(len(ds), size=2000, replace=False)
                true_samples = ds.data[idx]
                gen = model.sample(2000, args.sampling_steps, integrator="heun").numpy()
                vis_path = os.path.join(
                    args.save_dir, f"sample_pca_step_{step}{suffix}.png")
                visualize_samples_pca(true_samples, gen, save_path=vis_path,
                                      title_suffix=f" (step {step}{suffix})")
            model.train()

    model.eval()
    ckpt_path = os.path.join(args.save_dir, args.ckpt)
    torch.save({"model": model.model.state_dict()}, ckpt_path)
    print(f"[train_30d] checkpoint saved at {ckpt_path}")


if __name__ == "__main__":
    main()
