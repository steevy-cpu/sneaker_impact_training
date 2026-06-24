"""
train_brand.py -- local BRAND classifier (Goal B). Now at v2.

Still a from-scratch CNN, commented layer-by-layer so it stays fully explainable
(especially the hidden layers). v2 lifts v1's accuracy ceiling with three
standard, well-understood upgrades — none change the mental model:
  * BatchNorm after each conv (stable activations -> faster, higher-LR training)
  * stronger augmentation (flip/rotate/recolor/re-crop -> less overfitting)
  * a cosine learning-rate schedule + a 4th conv block (more capacity)
v1 (the bare 3-block net) is preserved in git history. Next on the roadmap:
v3 transfer learning, v4 fine-tune on human gold + color/model heads.

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
#  THE NEURAL NETWORK  (v2)
#  Same idea as v1 — a CNN = [feature extractor] + [classifier head] — but with
#  two upgrades that lift the v1 accuracy ceiling WITHOUT changing the story:
#
#    * BatchNorm after every conv. It normalizes that layer's outputs across the
#      batch to ~zero-mean/unit-variance (then learns a scale+shift). This keeps
#      the numbers feeding the next layer in a stable range as weights change,
#      so the net trains much faster, tolerates a higher learning rate, and gets
#      a little regularization for free. Order per block: Conv -> BN -> ReLU -> Pool.
#    * A 4th conv block (deeper = can learn more abstract brand motifs).
#
#  Reminder of the per-layer roles:
#    Conv2d  slides small 3x3 filters to detect LOCAL patterns; out_channels =
#            how many different pattern-detectors this layer learns.
#    ReLU    keeps only positives (max(0,x)) — the non-linearity that lets a deep
#            net model complex shapes, not just one straight-line function.
#    MaxPool halves H&W, keeping the strongest response per 2x2 patch (summarize
#            + small shift-invariance).
#  Going deeper: channels GROW (more feature types), spatial size SHRINKS.
#  edges/colors -> textures/curves -> parts (logos, stripes, midsole) -> motifs.
# ==========================================================================
class BrandCNN(nn.Module):
    def __init__(self, num_classes, img_size=128, use_bn=False):
        super().__init__()

        # Helper: one block = Conv -> [BatchNorm] -> ReLU -> MaxPool.
        # NOTE on BatchNorm: textbook-recommended, but EMPIRICALLY it hurt this
        # tiny from-scratch net on our noisy pseudo-labels — with BN the loss
        # crawled (~0.21 val) vs 0.44 without it, across several LRs/aug settings.
        # So it's OFF by default; the real, robust accuracy jump is transfer
        # learning (v3), where a pretrained backbone makes BN behave. Kept as a
        # toggle (--bn) for experimentation.
        def block(cin, cout):
            layers = [nn.Conv2d(cin, cout, kernel_size=3, padding=1)]  # detect patterns
            if use_bn:
                layers.append(nn.BatchNorm2d(cout))                   # stabilize activations
            layers += [nn.ReLU(inplace=True),                         # non-linearity
                       nn.MaxPool2d(2, 2)]                            # downsample 2x
            return nn.Sequential(*layers)

        # 3 blocks: 3 -> 16 -> 32 -> 64 feature maps. Same depth/width as the v1
        # baseline (which trained well) — v2's ONLY architecture change is the
        # BatchNorm inside each block. A deeper 4-block/256-ch version was tried
        # and trained far worse: too much capacity to optimize from scratch on
        # ~7k images, the loss just crawled. Lesson: add power carefully; the big
        # jump comes from transfer learning (v3), not from a bigger scratch net.
        self.features = nn.Sequential(
            block(3, 16),     # -> 16 x 64 x 64   (edges, colors)
            block(16, 32),    # -> 32 x 32 x 32   (textures, curves)
            block(32, 64),    # -> 64 x 16 x 16   (parts: logos, stripes, midsole)
        )

        # After 3 pools a 128px image is 128/2^3 = 16 px wide. Final map: 64 x 16 x 16.
        feat = img_size // 8                         # 128 // 8 = 16
        self.flat_dim = 64 * feat * feat             # 16384 for 128px input

        # Classifier head: flatten -> hidden FC (mix all features into 256
        # "brand-evidence" units) -> dropout -> output score per brand.
        self.classifier = nn.Sequential(
            nn.Flatten(),                            # (B, 256, 8, 8) -> (B, 16384)
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),                         # drop 40% of units (train only)
            nn.Linear(256, num_classes),             # -> one logit per brand
        )

    def forward(self, x):                # x: (B, 3, 128, 128)
        x = self.features(x)             # conv stack -> (B, 256, 8, 8)
        x = self.classifier(x)           # head        -> (B, num_classes) logits
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
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--aug", choices=["light", "heavy"], default="light")
    ap.add_argument("--bn", action="store_true", help="add BatchNorm (experimental — hurt v1 from scratch)")
    ap.add_argument("--arch", default="scratch",
                    help="'scratch' = the explainable from-scratch CNN; OR a timm "
                         "backbone for TRANSFER LEARNING (v3), e.g. resnet18, "
                         "mobilenetv3_small_100, efficientnet_b0")
    ap.add_argument("--lr", type=float, default=None,
                    help="default 1e-3 (scratch) / 3e-4 (transfer fine-tuning)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Transfer-learning vs scratch differ in input size, normalization and LR.
    transfer = args.arch != "scratch"
    img = 224 if transfer else args.img         # pretrained backbones expect 224px
    lr = args.lr if args.lr is not None else (3e-4 if transfer else 1e-3)
    classes, by_split, all_rows = load_rows(args.manifest, args.min_count)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    # Human-gold rows are precious GROUND TRUTH. NEVER train on them — hold them
    # ALL out as the clean (leak-free) evaluation set; we have plenty of silver
    # (cloud) labels to train on. Otherwise gold that lands in the train split
    # inflates the "gold accuracy" into meaningless training accuracy.
    n_before = len(by_split["train"])
    by_split["train"] = [r for r in by_split["train"] if r["label_quality"] != "human"]
    print(f"[train] held {n_before - len(by_split['train'])} human-gold rows OUT of train (eval-only)")
    print(f"[train] arch={args.arch}  img={img}  lr={lr}")
    print(f"[train] {len(classes)} brands: {classes}")
    print(f"[train] split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in by_split.items()))

    # Transforms. v2 augments harder: every epoch the net sees a slightly
    # different version of each crop (re-cropped, flipped, rotated, recolored),
    # so it learns the brand cue is invariant to those nuisances -> less
    # overfitting (the thing that capped v1). Eval is deterministic (no aug).
    # Pretrained backbones need ImageNet normalization + 224px; the scratch net
    # uses a simple [-1,1] scaling at 128px. Transfer learning tolerates (likes)
    # a bit of crop augmentation because the pretrained prior anchors it.
    if transfer:
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        norm = transforms.Normalize([0.5] * 3, [0.5] * 3)
    if args.aug == "heavy":
        crop = [transforms.RandomResizedCrop(img, scale=(0.7, 1.0)),
                transforms.RandomRotation(15), transforms.ColorJitter(0.2, 0.2, 0.2)]
    elif transfer:                                       # mild crop aug
        crop = [transforms.RandomResizedCrop(img, scale=(0.8, 1.0)),
                transforms.ColorJitter(0.1, 0.1, 0.1)]
    else:                                                # scratch: just resize
        crop = [transforms.Resize((img, img)), transforms.ColorJitter(0.1, 0.1, 0.1)]
    train_tfm = transforms.Compose(
        crop + [transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    eval_tfm = transforms.Compose([
        transforms.Resize((img, img)), transforms.ToTensor(), norm])

    def loader(split, tfm, shuffle):
        return DataLoader(ShoeBrandDataset(by_split[split], class_to_idx, tfm),
                          batch_size=args.batch, shuffle=shuffle,
                          num_workers=args.workers, pin_memory=(device == "cuda"))
    train_loader = loader("train", train_tfm, True)
    val_loader   = loader("val",   eval_tfm,  False)
    test_loader  = loader("test",  eval_tfm,  False)

    if transfer:
        import timm
        # TRANSFER LEARNING: load a backbone PRE-TRAINED on ImageNet (1.2M images).
        # It already learned generic visual features (edges -> textures -> shapes);
        # timm replaces its final classification layer with a fresh Linear -> our
        # K brands, and we FINE-TUNE the whole net at a low LR so it adapts to
        # shoes without forgetting that prior. This is why it converges fast and
        # far higher than a from-scratch net on only ~8k noisy images.
        model = timm.create_model(args.arch, pretrained=True, num_classes=len(classes)).to(device)
    else:
        model = BrandCNN(len(classes), img, use_bn=args.bn).to(device)

    # Class weights = inverse frequency, so the loss cares about a rare brand (On)
    # as much as a common one (Brooks) instead of just predicting the majority.
    tr_counts = Counter(r["make"] for r in by_split["train"])
    w = torch.tensor([len(by_split["train"]) / (len(classes) * tr_counts[c])
                      for c in classes], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)        # softmax + negative-log-likelihood
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine LR schedule: glide the learning rate from `lr` down to ~0 over all
    # epochs — fast, coarse learning early; gentle fine-settling late. Usually
    # squeezes out a few extra points vs a fixed LR. We call scheduler.step()
    # once per epoch (below).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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
        scheduler.step()                       # advance the cosine LR once per epoch

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
               "arch": args.arch, "img": img, "lr": lr, "config": vars(args)},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2)

    print(f"\n=== DONE -> {args.out}/ ===")
    print(f"  best val acc:  {best_val:.3f}")
    print(f"  TEST acc:      {test_acc:.3f}   (vs SILVER labels = agreement with Gemini, not truth)")
    print(f"  human-gold acc: {gold_acc if gold_acc is None else round(gold_acc,3)} "
          f"(n={len(gold_rows)} — the only real check; grows as labelers work)")


if __name__ == "__main__":
    main()
