"""
Train the Flow-Matching proposal q_theta on the 30D GMM target (cov = VAR*I_30).

Default 800k steps (matches the checkpoint shipped in results/fm_model.pt).
On an RTX 4090 this is ~80 steps/s, i.e. ~2.8 h.

  python train.py --steps 800000 --ckpt results/fm_model.pt --gpu 0
"""

import os
import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import FlowMatchingOT, set_seed
from gmm import GMMDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=800_000)
    p.add_argument("--ckpt", type=str, default="results/fm_model.pt")
    p.add_argument("--seed", type=int, default=53)
    p.add_argument("--num_samples", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_blocks", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--log_window", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} ckpt={args.ckpt} steps={args.steps}")

    ds = GMMDataset(num_samples=args.num_samples, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        pin_memory=(device.type == "cuda"), num_workers=0)

    model = FlowMatchingOT(
        dim=30, hidden_dim=args.hidden_dim, num_blocks=args.num_blocks,
        sigma=0.0, lr=args.lr, device=device,
        base_dist="logistic", base_loc=0.0, base_scale=1.0,
    )
    model.to(device)
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)

    it = iter(loader)
    model.train()
    t0 = time.time()
    losses = []
    for step in tqdm(range(1, args.steps + 1), desc="train"):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = batch.to(device, non_blocking=True)
        loss = model.forward(batch)
        model.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.optimizer.step()
        losses.append(loss.item())
        if step % args.log_window == 0:
            dt = time.time() - t0; t0 = time.time()
            tp = args.log_window / dt if dt > 0 else float('nan')
            print(f"[step {step}/{args.steps}] loss={np.mean(losses[-args.log_window:]):.4f} "
                  f"throughput={tp:.1f} steps/s", flush=True)

    model.eval()
    model.save(args.ckpt)
    print(f"[train] saved {args.ckpt}")


if __name__ == "__main__":
    main()
