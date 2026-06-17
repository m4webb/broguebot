"""Warm up ONLY the value head on top of a frozen BC policy.

The manual-teacher BC (runs/manual_bc2) clones navigation well (val ~0.80) but its
value head is random (the human/scripted-move data had no rewards). PPO from a
random value head produces noise advantages that DEGRADE the good policy (observed:
depth 1.35 BC -> 1.02 after PPO). Jointly training policy+value in BC instead
cripples the policy (value-loss gradients disrupt the shared encoder: val 0.25).

So: load the strong BC checkpoint, FREEZE everything except value_head, and fit the
value head to the depth-progress returns (relabeled into data/manual). Result: the
strong policy is untouched, and PPO starts with sane advantages.

  .venv/bin/python -m broguebot.nn.value_warmup --init runs/manual_bc2/bc_best.pt \
      --data data/manual --out runs/manual_bc2/bc_vh.pt --steps 800
"""

import argparse
import contextlib

import torch
import torch.nn.functional as F

from .model import BroguePolicy, Config
from .train_bc import StreamSampler
from .trajlog import episode_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="runs/manual_bc2/bc_best.pt")
    ap.add_argument("--data", default="data/manual")
    ap.add_argument("--out", default="runs/manual_bc2/bc_vh.pt")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    ckpt = torch.load(args.init, map_location=dev)
    model = BroguePolicy(getattr(Config, ckpt["config"])()).to(dev)
    model.load_state_dict(ckpt["model"])
    disabled = ckpt.get("disabled") or []
    if disabled:
        model.set_disabled_actions(disabled)
    model.grad_checkpoint = True
    model.encode_chunk = 128

    # freeze everything but the value head
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.value_head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.value_head.parameters(), lr=args.lr)

    files = episode_files(args.data, skip_truncated=True)
    sampler = StreamSampler(files, args.window, args.batch, gamma=args.gamma)
    carry = model.initial_state(args.batch, dev)

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) \
            if dev == "cuda" else contextlib.nullcontext()

    print(f"{len(files)} episodes; warming value head ({args.steps} steps), "
          f"disabled={disabled}")
    for step in range(1, args.steps + 1):
        obs, _acts, rets, reset = sampler.sample()
        obs = {k: v.to(dev) for k, v in obs.items()}
        rets = rets.to(dev)
        hidden = carry.detach() * (~reset).to(dev).float().unsqueeze(-1)
        with amp():
            _logits, values, carry = model.unroll(obs, hidden)
        loss = F.mse_loss(values.float(), rets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            v = values.float()
            print(f"vstep {step}: vloss {loss.item():.4f} "
                  f"pred[{v.min():.2f},{v.max():.2f}] tgt_mean {rets.mean():.2f}",
                  flush=True)
    torch.save({"model": model.state_dict(), "config": ckpt["config"],
                "disabled": disabled, "val_acc": ckpt.get("val_acc")}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
