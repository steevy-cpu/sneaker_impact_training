"""
validate_gate.py -- does the learned shoe-gate fix SAM2's over-detection while
keeping the recall win? Compares, on the hand-counted ground truth + a sample:
  YOLOE (current)  |  SAM2+size  |  SAM2+size+GATE
Reports detection count-recall AND over-detection (det - gt), plus pairs/slivers.
"""
import os, json, sqlite3, time
import cv2, torch, timm
import config, segment_utils as su
from pair_utils import pair_shoes_visual
from embedder_utils import build_image_embedder
from torchvision import transforms

config.SEGMENT_MODEL = "yoloe-11m-seg.pt"
TP = "../images/table_photos"; OUT = "/tmp/validate_gate"; os.makedirs(OUT, exist_ok=True)
DEV = su._resolve_device()
SAM_MAX = 1536; AF_LO, AF_HI, AR_MAX = 0.004, 0.12, 4.5
GTF = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224,224)),
    transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


def sam_masks(sam, image):
    h, w = image.shape[:2]
    s = SAM_MAX/float(max(h,w)) if max(h,w) > SAM_MAX else 1.0
    img_s = cv2.resize(image,(int(w*s),int(h*s))) if s!=1.0 else image
    inv=1.0/s; r=sam(img_s, verbose=False, device=DEV)[0]
    segs=[]
    if r.masks is None: return segs
    for p in r.masks.xy:
        if len(p)<3: continue
        x1,y1=int(p[:,0].min()*inv),int(p[:,1].min()*inv)
        x2,y2=int(p[:,0].max()*inv),int(p[:,1].max()*inv)
        bw,bh=x2-x1,y2-y1
        if bw<=0 or bh<=0: continue
        af=(bw*bh)/float(w*h); ar=max(bw,bh)/float(min(bw,bh))
        if af<AF_LO or af>AF_HI or ar>AR_MAX: continue
        segs.append(su.Segment((x1,y1,x2,y2),1.0,"shoe",None))
    return segs


@torch.no_grad()
def gate_keep(gate, image, segs):
    """Keep only masks the gate calls 'shoe' (class 1)."""
    if not segs: return segs
    h,w=image.shape[:2]; crops=[]
    for s in segs:
        x1,y1,x2,y2=s.bbox; crops.append(GTF(image[max(0,y1):y2, max(0,x1):x2]))
    x=torch.stack(crops).to(DEV)
    pred=gate(x).argmax(1).cpu().tolist()
    return [s for s,p in zip(segs,pred) if p==1]


def tiny(seg,w,h):
    x1,y1,x2,y2=seg.bbox; bw,bh=x2-x1,y2-y1
    if bw<=0 or bh<=0: return True
    return (bw*bh)/float(w*h)<0.004 or max(bw,bh)/float(min(bw,bh))>6.0


def main():
    detector=su.build_segmenter(config); embedder=build_image_embedder(config)
    from ultralytics import SAM; sam=SAM("sam2_b.pt")
    gate=timm.create_model("resnet18", pretrained=False, num_classes=2).to(DEV).eval()
    gate.load_state_dict(torch.load("dataset/gate_clf/gate_cnn.pt")); print("[val] gate loaded")

    def vpair(image,segs):
        return pair_shoes_visual(image,segs,embedder,
            spatial_weight=getattr(config,"SEGMENT_PAIR_SPATIAL_WEIGHT",0.15),
            min_sim=getattr(config,"SEGMENT_PAIR_MIN_SIM",0.5), log=lambda *_: None)

    # ---- Part A: recall + over-detection vs ground truth ----
    print("\n===== PART A: detection vs hand-counted gt (recall | over-detect) =====")
    gt=json.load(open("tiling_gt.json"))["photos"]
    R={"yo":[],"sz":[],"gt":[]}; O={"yo":0,"sz":0,"gt":0}
    print(f"{'photo':24} {'gt':>3} {'yoloe':>6} {'sam+sz':>7} {'sam+gate':>9}")
    for e in gt:
        img=cv2.imread(f"{TP}/{e['image']}")
        if img is None or e['gt_shoes']==0: continue
        g=e['gt_shoes']
        nyo=len(detector.segment(img))
        masks=sam_masks(sam,img); nsz=len(masks)
        ngt=len(gate_keep(gate,img,masks))
        for k,n in (("yo",nyo),("sz",nsz),("gt",ngt)):
            R[k].append(min(n,g)/g); O[k]+=max(0,n-g)
        print(f"{e['image']:24} {g:>3} {nyo:>6} {nsz:>7} {ngt:>9}")
    for k,lab in (("yo","YOLOE"),("sz","SAM2+size"),("gt","SAM2+gate")):
        if R[k]:
            print(f"  {lab:11} recall {sum(R[k])/len(R[k]):.3f}   over-detection (total) {O[k]}")

    # ---- Part B: pairs / slivers on a sample ----
    conn=sqlite3.connect("file:../sneakers.db?mode=ro",uri=True)
    rows=conn.execute("""SELECT id FROM table_photos WHERE status='completed'
        AND image_path IS NOT NULL AND num_pairs>0 ORDER BY num_pairs""").fetchall()
    ids=[rows[int(i*(len(rows)-1)/11)][0] for i in range(12)]; ids=list(dict.fromkeys(ids))
    print(f"\n===== PART B: pairs/slivers on {len(ids)} boxes (CUR vs SAM2+gate) =====")
    cp=npp=csl=nsl=more=0
    for k,pid in enumerate(ids):
        img=cv2.imread(f"{TP}/{pid}.jpg")
        if img is None: continue
        h,w=img.shape[:2]
        cur=vpair(img, detector.segment(img))
        new=vpair(img, gate_keep(gate,img,sam_masks(sam,img)))
        c=sum(1 for s in cur if getattr(s,"label","")=="pair")
        n=sum(1 for s in new if getattr(s,"label","")=="pair")
        cp+=c; npp+=n; more+= 1 if n>c else 0
        csl+=sum(1 for s in cur if tiny(s,w,h)); nsl+=sum(1 for s in new if tiny(s,w,h))
        if k<6:
            vis=img.copy()
            for s in new:
                col=(0,230,0) if getattr(s,"label","")=="pair" else (0,140,255)
                cv2.rectangle(vis,s.bbox[:2],s.bbox[2:],col,3)
            cv2.imwrite(f"{OUT}/{pid}_GATE.jpg", cv2.resize(vis,(1280,720)))
        print(f"  {pid:22} CUR {c:>2}p  GATE {n:>2}p")
    print(f"\nTOTALS: pairs CUR {cp} / GATE {npp} (gate more on {more}/{len(ids)})  "
          f"slivers CUR {csl} / GATE {nsl}")
    print(f"overlays -> {OUT}/   VALIDATE DONE")


if __name__ == "__main__":
    main()
