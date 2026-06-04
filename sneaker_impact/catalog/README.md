# CLIP catalog (Phase C verifier)

Drop reference sneaker images here, organized by **brand** then **model**:

```
catalog/
  Nike/
    Air Force 1/
      af1_white_1.jpg
      af1_white_2.jpg
    Dunk Low/
      dunk_panda_1.jpg
  Adidas/
    Samba/
      samba_1.jpg
```

- Folder level 1 = **brand**, level 2 = **model**, files = images of that model.
- Multiple images per model is good — more views = better matching.
- A public dataset (e.g. a Kaggle/HuggingFace sneaker set) can be reshaped into
  this layout and dropped in.

Then rebuild the index:

```bash
python build_catalog_index.py
```

This embeds every image here **plus** our confirmed `label_data/` with CLIP and
writes `sneaker_impact/clip_index.npz` (+ `.json`). The `clip-index` model
backend matches crops against it for a real similarity score + source link.

Images here are git-ignored (data); this README is kept.
