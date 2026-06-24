"""
train_brand.py -- v1 local BRAND classifier (Goal B).

A deliberately SIMPLE convolutional neural network, written from scratch and
commented layer-by-layer, so it can be fully explained (especially the hidden
layers). It learns to predict a shoe's brand from a top-down pair crop, so the
pipeline depends less on the cloud. Accuracy is secondary to clarity here; the
roadmap (BatchNorm -> augmentation -> transfer learning -> human-gold fine-tune)
adds power later WITHOUT changing the mental model below.

Reads the manifest from build_dataset.py, trains on the silver+gold TRAIN split,
early-stops on VAL, and reports TEST + human-gold accuracy. Offline + bounded
VRAM, so it never disturbs the live dashboard.

    cd sneaker_impact_training
    PYTHONPATH=. /usr/bin/python3 train_brand.py
    PYTHONPATH=. /usr/bin/python3 train_brand.py --epochs 15 --min-count 100
"""
import argparse
import csv
import json
import os
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

torch.manual_seed(42)                         # reproducible runs


# ==========================================================================
#  THE NEURAL NETWORK
#  A small CNN = [feature extractor] + [classifier head].
#
#  Feature extractor: a stack of "blocks", each = Conv -> ReLU -> MaxPool.
#    * Conv2d slides small 3x3 filters across the image to detect LOCAL patterns.
#      `out_channels` is how many different pattern-detectors this layer learns.
#    * ReLU keeps only positive activations (max(0,x)); this non-linearity is what
#      lets a deep net model complex shapes instead of just one linear function.
#    * MaxPool(2) halves height & width, keeping the strongest value in each 2x2
#      patch -> summarizes the layer and makes it a bit shift-invariant.
#  As we go deeper: channels GROW (more kinds of features) while the spatial size
#  SHRINKS. Early layers see edges/colors; deeper layers combine those into
#  textures, then object parts (logos, stripes, midsoles).
#
#  Classifier head: flatten the final feature maps into one vector and pass it
#  through fully-connected (Linear) layers that weigh the features into a score
#  per brand.
# ==========================================================================
class BrandCNN(nn.Module):
    def __init__(self, num_classes, img_size=128):
        super().__init__()

        # --- hidden conv layers (the feature extractor) ---
        # Block 1: 3 RGB channels -> 16 feature maps. Learns LOW-level cues:
        #          edges, corners, flat color regions. padding=1 keeps H,W the same
        #          before pooling so a 3x3 filter can sit on border pixels too.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # Block 2: 16 -> 32 maps. Combines edges into MID-level cues: textures, curves.
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Block 3: 32 -> 64 maps. Combines textures into PARTS: logo marks, stripes,
        #          the chunky midsole silhouette that often gives a brand away.
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)   # applied after every block

        # After 3 pools, img_size is halved 3x (128 -> 64 -> 32 -> 16). The final
        # feature map is therefore 64 channels x 16 x 16. Flattened = 64*16*16.
        feat = (img_size // 8)                      # 128 // 8 = 16
        self.flat_dim = 64 * feat * feat            # 16384 for 128px input

        # --- classifier head ---
        # Hidden fully-connected layer: mixes ALL 16384 feature numbers into 256
        # "brand-evidence" units. This is where the net reasons over the whole shoe
        # ("dark midsole + N-shaped panel -> New Balance-ish").
        self.fc1 = nn.Linear(self.flat_dim, 256)
        # Dropout zeroes 40% of those 256 units at random DURING TRAINING only, so
        # the net can't lean on any single feature -> reduces overfitting.
        self.dropout = nn.Dropout(0.4)
        # Output layer: 256 -> one raw score (logit) per brand. Softmax (applied
        # inside the loss) turns these into probabilities.
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):                       # x: (B, 3, 128, 128)
        x = self.pool(F.relu(self.conv1(x)))    # -> (B, 16, 64, 64)
        x = self.pool(F.relu(self.conv2(x)))    # -> (B, 32, 32, 32)
        x = self.pool(F.relu(self.conv3(x)))    # -> (B, 64, 16, 16)
        x = torch.flatten(x, start_dim=1)       # -> (B, 16384)  (keep batch dim)
        x = F.relu(self.fc1(x))                 # -> (B, 256)    hidden representation
        x = self.dropout(x)
        x = self.fc2(x)                         # -> (B, num_classes)  logits
        return x


# ==========================================================================
#  DATA
# ==========================================================================
class ShoeBrandDataset(Dataset):
    """Yields (image_tensor, label_index) for rows of one split whose brand is in
    the kept class list. Relative manifest paths resolve from the engine dir."""

    def __init__(self, rows, class_to_idx, tfm):
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.tfm = tfm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["filepath"]).convert("RGB")   # 3-channel, any size
        return self.tfm(img), self.class_to_idx[r["make"]]


def load_rows(manifest, min_count):
    rows = list(csv.DictReader(open(manifest)))
    counts = Counter(r["make"] for r in rows if r["make"] not in ("unknown", "other"))
    classes = sorted([m for m, n in counts.items() if n >= min_count])
    keep = set(classes)
    rows = [r for r in rows if r["make"] in keep and os.path.exists(r["filepath"])]
    by_split = {"train": [], "val": [], "test": []}
    for r in rows:
        by_split.get(r["split"], by_split["train"]).append(r)
    return classes, by_split, rows


# ==========================================================================
#  EVAL helper
# ==========================================================================
@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        pred = model(imgs).argmax(dim=1)        # the highest-scoring brand
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="dataset/manifest.csv")
    ap.add_argument("--out", default="dataset/brand_clf")
    ap.add_argument("--min-count", type=int, default=100, help="min images per brand to include")
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)       # small -> bounded VRAM
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes, by_split, all_rows = load_rows(args.manifest, args.min_count)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"[train] {len(classes)} brands: {classes}")
    print(f"[train] split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in by_split.items()))

    # Transforms: resize to a fixed square, (train) random horizontal flip for a
    # little augmentation, to tensor in [0,1], then normalize to ~[-1,1].
    norm = transforms.Normalize([0.5] * 3, [0.5] * 3)
    train_tfm = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm])
    eval_tfm = transforms.Compose([
        transforms.Resize((args.img, args.img)),
        transforms.ToTensor(), norm])

    def loader(split, tfm, shuffle):
        return DataLoader(ShoeBrandDataset(by_split[split], class_to_idx, tfm),
                          batch_size=args.batch, shuffle=shuffle,
                          num_workers=args.workers, pin_memory=(device == "cuda"))
    train_loader = loader("train", train_tfm, True)
    val_loader   = loader("val",   eval_tfm,  False)
    test_loader  = loader("test",  eval_tfm,  False)

    model = BrandCNN(len(classes), args.img).to(device)

    # Class weights = inverse frequency, so the loss cares about a rare brand (On)
    # as much as a common one (Brooks) instead of just predicting the majority.
    tr_counts = Counter(r["make"] for r in by_split["train"])
    w = torch.tensor([len(by_split["train"]) / (len(classes) * tr_counts[c])
                      for c in classes], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)        # softmax + negative-log-likelihood
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out, exist_ok=True)
    best_val, history = 0.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # --- the canonical 5-step training step ---
            optimizer.zero_grad()              # 1) clear gradients from last step
            logits = model(imgs)               # 2) forward pass -> brand scores
            loss = criterion(logits, labels)   # 3) how wrong were we?
            loss.backward()                    # 4) backprop -> gradient for every weight
            optimizer.step()                   # 5) nudge weights to lower the loss
            run_loss += loss.item() * labels.size(0); seen += labels.size(0)
        val_acc = accuracy(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": run_loss / seen, "val_acc": val_acc})
        print(f"  epoch {epoch:2d}  loss {run_loss/seen:.3f}  val_acc {val_acc:.3f}  "
              f"({time.time()-t0:.0f}s)")
        if val_acc >= best_val:                # keep the best-on-validation weights
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(args.out, "brand_cnn.pt"))

    # Reload best and report final numbers.
    model.load_state_dict(torch.load(os.path.join(args.out, "brand_cnn.pt")))
    test_acc = accuracy(model, test_loader, device)
    # Human-gold rows are the only TRUE labels; report separately (tiny for now).
    gold_rows = [r for r in all_rows if r["label_quality"] == "human"]
    gold_acc = None
    if gold_rows:
        gl = DataLoader(ShoeBrandDataset(gold_rows, class_to_idx, eval_tfm), batch_size=args.batch)
        gold_acc = accuracy(model, gl, device)

    json.dump({c: i for c, i in class_to_idx.items()},
              open(os.path.join(args.out, "classes.json"), "w"), indent=2)
    json.dump({"classes": classes, "best_val_acc": best_val, "test_acc": test_acc,
               "gold_acc": gold_acc, "gold_n": len(gold_rows), "history": history,
               "config": vars(args)},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2)

    print(f"\n=== DONE -> {args.out}/ ===")
    print(f"  best val acc:  {best_val:.3f}")
    print(f"  TEST acc:      {test_acc:.3f}   (vs SILVER labels = agreement with Gemini, not truth)")
    print(f"  human-gold acc: {gold_acc if gold_acc is None else round(gold_acc,3)} "
          f"(n={len(gold_rows)} — the only real check; grows as labelers work)")


if __name__ == "__main__":
    main()
