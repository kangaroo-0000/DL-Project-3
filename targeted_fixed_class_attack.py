import argparse, json, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models

@torch.no_grad()
def evaluate(model, loader, idx2true, device, tag=""):
    top1 = top5 = tot = 0
    for x, labs in loader:
        x = x.to(device, non_blocking=True)
        labs = torch.tensor([idx2true[int(l)] for l in labs], device=device)
        logits = model(x)
        top1 += (logits.argmax(1) == labs).sum().item()
        top5 += (logits.topk(5, 1)[1] == labs[:, None]).any(1).sum().item()
        tot  += labs.size(0)
    print(f"{tag:<8}Top‑1 {top1/tot*100:6.2f}%   Top‑5 {top5/tot*100:6.2f}%")
    return top1 / tot, top5 / tot

def targeted_pgd(model, x, target, eps_n, alpha_n, steps, min_v, max_v, rand_start):
    if rand_start:
        x_adv = x + torch.empty_like(x).uniform_(-1, 1) * eps_n
        x_adv = torch.clamp(x_adv, min_v, max_v)
    else:
        x_adv = x.clone()

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x_adv), target)
        loss.backward()
        grad_sign = x_adv.grad.sign()
        x_adv = x_adv - alpha_n * grad_sign
        x_adv = torch.clamp(x_adv, x - eps_n, x + eps_n)
        x_adv = torch.clamp(x_adv, min_v, max_v)

    return x_adv.detach()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="./TestDataSet")
    parser.add_argument("--labels", default="./TestDataSet/labels_list.json")
    parser.add_argument("--eps",    type=float, default=0.02)
    parser.add_argument("--steps",  type=int,   default=10)
    parser.add_argument("--batch",  type=int,   default=32)
    parser.add_argument("--workers",type=int,   default=4)
    parser.add_argument("--rand_start", action="store_true")
    parser.add_argument("--target_class", type=int, default=859, help="Target class index (default: toaster)")
    parser.add_argument("--out",    default="PGD-Targeted-Fixed-Step-10.pt")
    args = parser.parse_args()
    args.alpha = args.eps / args.steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required.")
    device = torch.device("cuda")

    torch.manual_seed(0); random.seed(0); np.random.seed(0)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalize = transforms.Normalize(mean.tolist(), std.tolist())

    eps_n   = torch.tensor(args.eps   / std, device=device).view(3,1,1)
    alpha_n = torch.tensor(args.alpha / std, device=device).view(3,1,1)
    min_val = torch.tensor(((0 - mean) / std).reshape(3,1,1), device=device)
    max_val = torch.tensor(((1 - mean) / std).reshape(3,1,1), device=device)
    std_tensor = torch.tensor(std, device=device).view(3,1,1)

    transform = transforms.Compose([transforms.ToTensor(), normalize])
    dataset   = datasets.ImageFolder(root=args.data, transform=transform)
    loader    = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    with open(args.labels) as f:
        idx2true = {i: int(e.split(":",1)[0]) for i, e in enumerate(json.load(f))}

    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
    model.eval()

    evaluate(model, loader, idx2true, device, "Clean")

    fixed_target_class = args.target_class
    adv_batches, lab_batches = [], []
    for x, labs in loader:
        x = x.to(device, non_blocking=True)
        y_true = torch.tensor([idx2true[int(l)] for l in labs], device=device)

        y_target = torch.full_like(y_true, fill_value=fixed_target_class)

        adv = targeted_pgd(model, x, y_target, eps_n, alpha_n, args.steps, min_val, max_val, args.rand_start)

        raw_delta = ((adv - x).abs() * std_tensor).max().item()
        assert raw_delta <= args.eps + 1e-6, f"ε‑constraint {raw_delta:.6f} > ε"

        adv_batches.append(adv)
        lab_batches.append(labs.to(device))

    adv_tensor = torch.cat(adv_batches)
    lab_tensor = torch.cat(lab_batches)

    pgd_loader = DataLoader(TensorDataset(adv_tensor, lab_tensor), batch_size=args.batch)
    evaluate(model, pgd_loader, idx2true, device, "Targeted-PGD")

    torch.save({"images": adv_tensor.cpu(), "labels": lab_tensor.cpu()}, args.out)
    print(f"Saved Targeted Adversarial Set → {args.out}")

if __name__ == "__main__":
    main()
