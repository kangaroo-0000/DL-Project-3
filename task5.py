"""
Example:
    python task5.py \
        --data    TestDataSet \
        --labels  TestDataSet/labels_list.json \
        --clean   Clean                     \
        --fgsm    FGSM.pt \
        --pgd     PGD-Step-10.pt \
        --patch   PGD-Step-5.pt \
        --patch2  PGD-Step-15.pt \
        --model   densenet121
"""

import argparse, json, torch, numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models

# ───────── helper ───────── #
@torch.no_grad()
def accuracy(model, loader, idx2true, device):
    t1 = t5 = n = 0
    for x, labs in loader:
        x = x.to(device, non_blocking=True)
        y = torch.tensor([idx2true[int(l)] for l in labs], device=device)
        out = model(x)
        t1 += (out.argmax(1) == y).sum().item()
        t5 += (out.topk(5, 1)[1] == y[:, None]).any(1).sum().item()
        n  += y.size(0)
    return 100 * t1 / n, 100 * t5 / n

# ───────── main ───────── #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--clean",  default="Clean",
                   help="keyword to print clean accuracy row")
    p.add_argument("--fgsm",   required=True)
    p.add_argument("--pgd",    required=True)
    p.add_argument("--patch",  required=True)
    p.add_argument("--patch2", required=True,
                   help="soft‑mask patch set (Task 4)")
    p.add_argument("--model",  default="densenet121",
                   help="torchvision.models attribute name")
    p.add_argument("--batch",  type=int, default=32)
    p.add_argument("--workers",type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # normalisation
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    tfm  = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean.tolist(), std.tolist())])

    # clean loader
    clean_ds = datasets.ImageFolder(args.data, transform=tfm)
    clean_ld = DataLoader(clean_ds, batch_size=args.batch, num_workers=args.workers)

    # folder‑idx → true ImageNet idx
    with open(args.labels) as f:
        idx2true = {i: int(e.split(":",1)[0]) for i, e in enumerate(json.load(f))}

    # helper to open .pt adversarial sets
    def adv_loader(path):
        ckpt = torch.load(path, map_location="cpu")
        return DataLoader(TensorDataset(ckpt["images"], ckpt["labels"]),
                          batch_size=args.batch, num_workers=args.workers)

    loaders = {
        "Clean":   clean_ld,
        "FGSM":    adv_loader(args.fgsm),
        "PGD":     adv_loader(args.pgd),
        "Patch":   adv_loader(args.patch),
        "Patch*":  adv_loader(args.patch2),
    }

    # model
    mdl_ctor = getattr(models, args.model)
    mdl = mdl_ctor(weights="IMAGENET1K_V1").to(device) 
    mdl.eval()

    print(f"\nTransfer accuracy on {args.model} ({args.batch}-batch)")
    print("Dataset   Top‑1   Top‑5")
    for name, ld in loaders.items():
        a1, a5 = accuracy(mdl, ld, idx2true, device)
        print(f"{name:<7}  {a1:6.2f}% {a5:6.2f}%")

if __name__ == "__main__":
    main()
