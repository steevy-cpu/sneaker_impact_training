"""
train_gate.py -- a binary shoe / not-shoe classifier (the SAM2 gate).

Transfer learning (resnet18) on the auto-labeled gate_data/. The job is coarse
(shoe vs reflection/box/rail/glass), so a pretrained net separates it easily even
with few negatives. Class-weighted because not-shoe is rare.

We report BOTH per-class recalls, which is what matters for the gate:
  * not-shoe recall = how much junk we CATCH (drop)
  * shoe recall     = how many real shoes we KEEP (don't wrongly drop)
The earlier YOLO gate failed exactly here: it caught junk but also dropped shoes.

Output: dataset/gate_clf/gate_cnn.pt + meta.json
"""
import os, glob, json, random, time
import torch, torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

random.seed(42); torch.manual_seed(42)
DATA = "dataset/gate_data"; OUT = "dataset/gate_clf"; os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# classes: 0 = notshoe, 1 = shoe
NORM = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
TRAIN_TF = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.15,0.15,0.15),
    transforms.ToTensor(), NORM])
EVAL_TF = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), NORM])


class DS(Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, y = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), y


def load():
    pos = glob.glob(f"{DATA}/shoe/*.jpg"); neg = glob.glob(f"{DATA}/notshoe/*.jpg")
    random.shuffle(pos); random.shuffle(neg)
    pos = pos[:600]                                   # cap to ease imbalance (still ~5:1)
    def split(lst):
        k = max(1, int(0.2*len(lst))); return lst[k:], lst[:k]   # train, val
    ptr, pva = split(pos); ntr, nva = split(neg)
    train = [(p,1) for p in ptr] + [(p,0) for p in ntr]
    val   = [(p,1) for p in pva] + [(p,0) for p in nva]
    random.shuffle(train)
    return train, val, len(ntr), len(ptr)


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); tp=tn=fp=fn=0
    for x,y in loader:
        x=x.to(DEV); pred=model(x).argmax(1).cpu()
        for pr,gt in zip(pred,y):
            if gt==1 and pr==1: tp+=1
            elif gt==1: fn+=1
            elif gt==0 and pr==0: tn+=1
            else: fp+=1
    shoe_rec = tp/(tp+fn) if tp+fn else 0
    notshoe_rec = tn/(tn+fp) if tn+fp else 0
    acc = (tp+tn)/max(1,tp+tn+fp+fn)
    return acc, shoe_rec, notshoe_rec, (tp,tn,fp,fn)


def main():
    train, val, n_neg_tr, n_pos_tr = load()
    print(f"[gate] train={len(train)} (pos {n_pos_tr}/neg {n_neg_tr})  val={len(val)}")
    tl = DataLoader(DS(train, TRAIN_TF), batch_size=32, shuffle=True, num_workers=4)
    vl = DataLoader(DS(val, EVAL_TF), batch_size=32, num_workers=4)
    model = timm.create_model("resnet18", pretrained=True, num_classes=2).to(DEV)
    # weight not-shoe higher (it's rare) so the net actually learns to catch junk
    w = torch.tensor([n_pos_tr/max(1,n_neg_tr), 1.0], dtype=torch.float32, device=DEV)
    crit = nn.CrossEntropyLoss(weight=w); opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    best=0
    for ep in range(1, 13):
        model.train(); t=time.time()
        for x,y in tl:
            x,y=x.to(DEV),y.to(DEV); opt.zero_grad()
            nn.functional.cross_entropy(model(x), y, weight=w).backward(); opt.step()
        acc, sr, nr, cm = evaluate(model, vl)
        print(f"  ep{ep:2d}  val_acc {acc:.3f}  shoe_recall {sr:.3f}  notshoe_recall {nr:.3f}  ({time.time()-t:.0f}s)")
        score = min(sr, nr)                           # want BOTH high
        if score >= best:
            best = score; torch.save(model.state_dict(), f"{OUT}/gate_cnn.pt")
    acc, sr, nr, cm = evaluate(model, vl)
    json.dump({"classes":{"notshoe":0,"shoe":1}, "val_acc":acc, "shoe_recall":sr,
               "notshoe_recall":nr, "confusion_tp_tn_fp_fn":cm, "arch":"resnet18"},
              open(f"{OUT}/meta.json","w"), indent=2)
    print(f"\n[gate] DONE -> {OUT}/  best min(recall)={best:.3f}")
    print(f"  final: shoe_recall {sr:.3f} (keep shoes)  notshoe_recall {nr:.3f} (catch junk)")


if __name__ == "__main__":
    main()
