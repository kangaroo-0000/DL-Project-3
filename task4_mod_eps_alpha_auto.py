#!/usr/bin/env python3
"""
Task 4 – 32 × 32 *soft*‑patch targeted PGD on ResNet‑34
• Blurred alpha mask (σ≈6) hides the hard square edge
• ε = 0.3, 80 steps, α = 0.006  →   subtle colour shift but strong attack
• Saves “Adversarial Test Set 3” ⟶  adv_set_patch_soft.pt
"""

import argparse, json, random, numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models

# ─────────────────────────── accuracy helper ──────────────────────────── #
@torch.no_grad()
def evaluate(model, loader, idx2true, device, tag=""):
    t1 = t5 = n = 0
    for x, labs in loader:
        x = x.to(device)
        labs = torch.tensor([idx2true[int(l)] for l in labs], device=device)
        logits = model(x)
        t1 += (logits.argmax(1) == labs).sum().item()
        t5 += (logits.topk(5, 1)[1] == labs[:, None]).any(1).sum().item()
        n += labs.size(0)
    print(f"{tag:<10}Top‑1 {t1/n*100:6.2f}%   Top‑5 {t5/n*100:6.2f}%")

# ────────────────────────── Gaussian blur mask ────────────────────────── #
def gaussian_blur(mask, k=15, sigma=6):
    """Separable 2‑D Gaussian blur for (B,1,H,W) masks."""
    coords = torch.arange(k, dtype=torch.float32, device=mask.device) - (k - 1) / 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, 1, k)
    mask = F.conv2d(mask, kernel, padding=(0, k // 2), groups=1)
    mask = F.conv2d(mask, kernel.transpose(2, 3), padding=(k // 2, 0), groups=1)
    return mask

# ───────────────────────────── patch‑PGD ───────────────────────────────── #
def patch_pgd(model, x, tgt, mask, eps_n, alpha_n, steps, min_v, max_v):
    x_adv = x.clone()
    mom = torch.zeros_like(x)          # momentum term
    beta = 0.9
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), tgt)     # targeted
        loss.backward()

        grad = x_adv.grad.sign() * mask
        mom = beta * mom + grad
        x_adv = x_adv - alpha_n * mom.sign()          # minimize toward tgt
        x_adv = torch.max(torch.min(x_adv, x + eps_n * mask), x - eps_n * mask)
        x_adv = torch.clamp(x_adv, min_v, max_v)
    return x_adv.detach()

# ─────────────────────────────── main ─────────────────────────────────── #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="./TestDataSet")
    p.add_argument("--labels", default="./TestDataSet/labels_list.json")
    p.add_argument("--eps",    type=float, default=0.3)   # smaller ε for subtlety
    p.add_argument("--steps",  type=int,   default=80)
    p.add_argument("--alpha",  type=float, default=0.006)
    p.add_argument("--batch",  type=int,   default=16)
    p.add_argument("--workers",type=int,   default=4)
    p.add_argument("--target", type=int,   default=0)     # tench
    p.add_argument("--out",    default="adv_set_patch_soft.pt")
    args = p.parse_args()
    args.alpha = args.eps / args.steps 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0); random.seed(0); np.random.seed(0)

    # ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std  = np.array([0.229, 0.224, 0.225], np.float32)
    normalize = transforms.Normalize(mean.tolist(), std.tolist())

    eps_n   = torch.tensor(args.eps   / std, device=device).view(3,1,1)
    alpha_n = torch.tensor(args.alpha / std, device=device).view(3,1,1)
    min_v   = torch.tensor(((0 - mean)/std).reshape(3,1,1), device=device)
    max_v   = torch.tensor(((1 - mean)/std).reshape(3,1,1), device=device)
    std_t   = torch.tensor(std, device=device).view(3,1,1)

    # data
    ds = datasets.ImageFolder(args.data,
          transform=transforms.Compose([transforms.ToTensor(), normalize]))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    with open(args.labels) as f:
        idx2true = {i: int(e.split(":",1)[0]) for i, e in enumerate(json.load(f))}

    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
    model.eval()

    evaluate(model, loader, idx2true, device, "Clean")

    adv_batches, lab_batches = [], []
    for x, labs in loader:
        x = x.to(device)
        B, C, H, W = x.shape

        # --- soft 32×32 mask per image ---
        hard = torch.zeros(B, 1, H, W, device=device)
        for i in range(B):
            top  = random.randint(0, H - 32)
            left = random.randint(0, W - 32)
            hard[i, :, top:top+32, left:left+32] = 1.0
        mask = gaussian_blur(hard, k=15, sigma=6).clamp(0, 1)
        mask = mask.repeat(1, 3, 1, 1)  # (B,3,H,W)

        tgt = torch.full((B,), args.target, device=device, dtype=torch.long)

        adv = patch_pgd(model, x, tgt, mask, eps_n, alpha_n,
                        args.steps, min_v, max_v)

        # ε‑check only where mask>0.5
        raw = ((adv - x).abs() * std_t * (mask > 0.5)).max().item()
        assert raw <= args.eps + 1e-6

        adv_batches.append(adv)
        lab_batches.append(labs.to(device))

    adv_tensor = torch.cat(adv_batches)
    lab_tensor = torch.cat(lab_batches)

    evaluate(model,
             DataLoader(TensorDataset(adv_tensor, lab_tensor), batch_size=args.batch),
             idx2true, device, "Patch‑PGD")

    torch.save({"images": adv_tensor.cpu(), "labels": lab_tensor.cpu()}, args.out)
    print(f"\nSaved Adversarial Test Set 3  →  {args.out}")

if __name__ == "__main__":
    main()
