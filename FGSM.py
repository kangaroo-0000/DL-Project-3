#!/usr/bin/env python3
"""
FGSM attack on TorchVision ResNet‑34 (Tasks 1 & 2) — GPU‑only version.

• Computes clean top‑1 / top‑5 accuracy on the 100‑class subset.
• Crafts ε‑bounded (raw‑pixel) adversarial examples with FGSM.
• Re‑evaluates accuracy and saves the perturbed tensor.

Example:
    python fgsm_resnet34_cuda.py --eps 0.02 --batch 32 \
        --data   ./TestDataSet \
        --labels ./TestDataSet/labels_list.json \
        --out    adversarial_set.pt
"""

import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models


# --------------------------------------------------------------------------- #
#                         helper: evaluation on a loader                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, idx_to_true, device, title=""):
    top1 = top5 = tot = 0
    for imgs, labs in loader:
        imgs = imgs.to(device, non_blocking=True)
        labs = torch.tensor([idx_to_true[int(l)] for l in labs], device=device)

        logits = model(imgs)
        pred1  = logits.argmax(1)
        _, p5  = logits.topk(5, dim=1)

        top1 += (pred1 == labs).sum().item()
        top5 += (p5 == labs.unsqueeze(1)).any(dim=1).sum().item()
        tot  += labs.size(0)

    print(f"{title:<8} Top‑1 {top1/tot*100:6.2f}%   Top‑5 {top5/tot*100:6.2f}%")
    return top1 / tot, top5 / tot


# --------------------------------------------------------------------------- #
#                       helper: one FGSM step on a batch                      #
# --------------------------------------------------------------------------- #
def fgsm_batch(model, imgs, true_lbls, eps_norm, min_val, max_val):
    imgs.requires_grad_(True)
    logits = model(imgs)
    F.cross_entropy(logits, true_lbls).backward()

    adv = imgs + eps_norm * imgs.grad.data.sign()
    adv = torch.max(torch.min(adv, max_val), min_val)
    return adv.detach()


# --------------------------------------------------------------------------- #
#                                    main                                     #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default="TestDataSet")
    ap.add_argument("--labels", default="TestDataSet/labels_list.json")
    ap.add_argument("--eps",    type=float, default=0.02,
                    help="ε in raw‑pixel units (0–1 scale)")
    ap.add_argument("--batch",  type=int,   default=32)
    ap.add_argument("--out",    default="adversarial_set.pt")
    ap.add_argument("--workers",type=int,   default=4)
    args = ap.parse_args()

    # ----- CUDA only -----
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for this script.")
    device = torch.device("cuda")
    torch.manual_seed(0)

    # ----- ImageNet normalisation -----
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalize = transforms.Normalize(mean.tolist(), std.tolist())

    eps_norm = torch.tensor(args.eps / std, device=device, dtype=torch.float32)[:, None, None]
    min_val  = torch.tensor(((0 - mean) / std).reshape(3, 1, 1),
                            device=device, dtype=torch.float32)
    max_val  = torch.tensor(((1 - mean) / std).reshape(3, 1, 1),
                            device=device, dtype=torch.float32)

    # ----- dataset -----
    ds = datasets.ImageFolder(args.data,
          transform=transforms.Compose([transforms.ToTensor(), normalize]))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers)

    # folder‑index → true ImageNet index
    with open(args.labels) as f:
        idx_to_true = {i: int(e.split(":",1)[0]) for i, e in enumerate(json.load(f))}

    # ----- model -----
    model = models.resnet34(
        weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
    model.eval()

    # 1. clean accuracy
    evaluate(model, loader, idx_to_true, device, "Clean")

    # 2. craft FGSM adversarial set
    adv_batches, lab_batches = [], []
    for imgs, labs in loader:
        imgs = imgs.to(device, non_blocking=True)
        true = torch.tensor([idx_to_true[int(l)] for l in labs], device=device)

        adv = fgsm_batch(model, imgs, true, eps_norm, min_val, max_val)

        # per‑batch ε check (raw pixels)
        raw_delta = (adv - imgs).abs() * torch.tensor(std, device=device).view(3,1,1)
        assert raw_delta.max().item() <= args.eps + 1e-6

        adv_batches.append(adv)           # keep on GPU for now
        lab_batches.append(labs.to(device))

    adv_tensor = torch.cat(adv_batches)   # still float32 on GPU
    lab_tensor = torch.cat(lab_batches)

    # 3. adversarial accuracy
    adv_loader = DataLoader(
        TensorDataset(adv_tensor, lab_tensor), batch_size=args.batch)
    evaluate(model, adv_loader, idx_to_true, device, "FGSM")

    # 4. save (move once to CPU)
    torch.save(
        {"images": adv_tensor.cpu(), "labels": lab_tensor.cpu()},
        args.out
    )
    print(f"\nSaved adversarial tensor → {args.out}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
