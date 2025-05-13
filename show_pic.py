import argparse, json, random, numpy as np, torch, matplotlib
matplotlib.use("Agg")                 
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models

# ------------------------ CLI ------------------------ #
ap = argparse.ArgumentParser()
ap.add_argument("--data",    required=True)
ap.add_argument("--labels",  required=True)
ap.add_argument("--adv_pt",  required=True)
ap.add_argument("--num",     type=int, default=5)
ap.add_argument("--outfile", default="adv_examples.png")
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------- ImageNet normalisation --------------- #
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
normalize  = transforms.Normalize(mean.tolist(), std.tolist())
inv_normal = transforms.Normalize((-mean/std).tolist(), (1/std).tolist())

# -------------- load clean & adv tensors ------------- #
clean_ds = datasets.ImageFolder(
    args.data, transform=transforms.Compose([transforms.ToTensor(), normalize])
)
clean_imgs, _ = next(iter(torch.utils.data.DataLoader(clean_ds, batch_size=len(clean_ds))))
adv_ckpt = torch.load(args.adv_pt, map_location="cpu")
adv_imgs = adv_ckpt["images"]          # tensor (500,C,H,W)

assert adv_imgs.shape == clean_imgs.shape, "Shape mismatch"

# -------------- label mapping & predictions ---------- #
with open(args.labels) as f:
    idx2true = {i: int(e.split(":", 1)[0]) for i, e in enumerate(json.load(f))}
names = models.ResNet34_Weights.IMAGENET1K_V1.meta["categories"]

model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
model.eval()

def pred(tensor):
    with torch.no_grad():
        return model(tensor.to(device)).argmax(1).cpu()

clean_pred = pred(clean_imgs)
adv_pred   = pred(adv_imgs)

# -------------- pick flipped examples ---------------- #
flip_idx = [i for i in range(len(clean_ds))
            if clean_pred[i] == idx2true[int(adv_ckpt["labels"][i])]
            and adv_pred[i]   != idx2true[int(adv_ckpt["labels"][i])]]

if not flip_idx:
    flip_idx = random.sample(range(len(clean_ds)), args.num)

picked = random.sample(flip_idx, min(args.num, len(flip_idx)))

# -------------- create figure & save ----------------- #
n = len(picked)
fig, ax = plt.subplots(2, n, figsize=(3*n, 6))

for col, i in enumerate(picked):
    true_id = idx2true[int(adv_ckpt["labels"][i])]
    clean_img = inv_normal(clean_imgs[i]).clamp(0,1).permute(1,2,0).numpy()
    adv_img   = inv_normal(adv_imgs[i]).clamp(0,1).permute(1,2,0).numpy()

    ax[0, col].imshow(clean_img); ax[0,col].axis("off")
    ax[1, col].imshow(adv_img);   ax[1,col].axis("off")

    ax[0, col].set_title(f"clean\ntrue={names[true_id]}\npred={names[clean_pred[i]]}",
                         fontsize=8)
    ax[1, col].set_title(f"adv\npred={names[adv_pred[i]]}", fontsize=8)

plt.tight_layout()
plt.savefig(args.outfile, dpi=200, bbox_inches="tight")
print("Saved figure →", args.outfile)
