"""
Generator for the Colab training notebook (smile_classifier_training.ipynb).

This script BUILDS the notebook using nbformat so the .ipynb is always valid.
Run:  python build_notebook.py
Output: smile_classifier_training.ipynb  (upload this to Google Colab)

Keeping the generator in the repo lets us regenerate/edit the notebook cleanly
instead of hand-editing fragile JSON.
"""
import nbformat as nbf

C = []
def md(s):   C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): C.append(nbf.v4.new_code_cell(s.strip("\n")))

# ─────────────────────────────────────────────────────────────────────────────
md(r"""
# 😊 AI Smile Classifier — Full Training Pipeline (CelebA + MobileNetV2)

**Run this in Google Colab.**  `Runtime → Change runtime type → GPU`, then run every
cell top-to-bottom. At the end you get **`smile_classifier.keras`** — download it and
place it in your project at `backend/model/smile_classifier.keras`.

This one notebook covers roadmap **Phases 2 → 12**:
Download → Analysis → Cleaning → Split → Preprocess → Augment → MobileNetV2
Transfer Learning → Initial Training → Fine-Tuning → Evaluation → Save.

The backend (FastAPI) and frontend (React) are built later on your laptop — not here.
""")

# ── Phase 0: setup ───────────────────────────────────────────────────────────
md("## Phase 0 — Setup\nColab already has TensorFlow, NumPy, pandas, matplotlib, scikit-learn, Pillow.\nWe only add `kagglehub` for the dataset download.")
code("!pip install -q kagglehub")

code(r"""
import os, glob, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("GPU:", gpus if gpus else "❌ NO GPU — go to Runtime > Change runtime type > GPU")
""")

md("### Configuration\nAll the knobs in one place. Increase `PER_CLASS` for higher accuracy (slower).")
code(r"""
IMG_SIZE   = 224          # MobileNetV2 input size
BATCH_SIZE = 32
PER_CLASS  = 15000        # images PER class in our balanced subset (30k total).
                          # Set higher (e.g. 40000) for better accuracy, or lower for speed.
AUTOTUNE   = tf.data.AUTOTUNE

INITIAL_EPOCHS  = 5       # head saturates fast with the base frozen — don't waste epochs here
FINETUNE_EPOCHS = 18      # fine-tuning is where accuracy grows -> 23 epochs total
                          # (EarlyStopping trims automatically if val_loss stops improving)
INITIAL_LR      = 1e-3
FINETUNE_LR     = 1e-5    # much smaller LR for fine-tuning (don't wreck pretrained weights)
FINETUNE_AT     = 100     # unfreeze MobileNetV2 layers from this index onward

CLASS_NAMES = ["not_smiling", "smiling"]   # 0 = not_smiling, 1 = smiling
MODEL_OUT   = "smile_classifier.keras"
""")

# ── Phase 1: Kaggle auth ─────────────────────────────────────────────────────
md(r"""
## Phase 1 — Kaggle authentication
You already used `kagglehub` in your BUSI project, so this probably works as-is.
If you get an auth error, uncomment ONE option below.
""")
code(r"""
import os
# Option A — interactive:
# import kagglehub; kagglehub.login()

# Option B — from kaggle.json (Kaggle > Settings > API > Create New Token):
# os.environ["KAGGLE_USERNAME"] = "your_username"
# os.environ["KAGGLE_KEY"]      = "your_key"
""")

# ── Phase 2: Download ────────────────────────────────────────────────────────
md("## Phase 2 — Download CelebA (~1.4 GB, one-time per session)")
code(r"""
import kagglehub
DATA_ROOT = kagglehub.dataset_download("jessicali9530/celeba-dataset")
print("Downloaded to:", DATA_ROOT)
""")

md("### Locate the files (robust to folder nesting)")
code(r"""
# Attributes CSV (has the 'Smiling' column)
csv_hits = glob.glob(os.path.join(DATA_ROOT, "**", "list_attr_celeba.csv"), recursive=True)
assert csv_hits, "list_attr_celeba.csv not found — check the download."
ATTR_CSV = csv_hits[0]

# Image directory = the folder holding the most .jpg files
jpg_by_dir = {}
for root, _d, files in os.walk(DATA_ROOT):
    n = sum(1 for f in files if f.lower().endswith(".jpg"))
    if n: jpg_by_dir[root] = n
IMG_DIR = max(jpg_by_dir, key=jpg_by_dir.get)

print("Attributes CSV:", ATTR_CSV)
print("Image dir     :", IMG_DIR, "->", jpg_by_dir[IMG_DIR], "jpgs")
""")

# ── Phase 3: labels + analysis ───────────────────────────────────────────────
md("## Phase 3 — Load attributes, build the smile label, analyze the data")
code(r"""
df = pd.read_csv(ATTR_CSV)
if "image_id" not in df.columns:                 # normalize first column name
    df = df.rename(columns={df.columns[0]: "image_id"})
assert "Smiling" in df.columns, f"'Smiling' missing. Got: {list(df.columns)[:5]}"

# CelebA attributes are -1 / +1.  Smiling == 1 -> smiling(1), else not_smiling(0)
df["label"] = (df["Smiling"] == 1).astype(int)
df["path"]  = df["image_id"].apply(lambda f: os.path.join(IMG_DIR, f))
print(df[["image_id", "Smiling", "label"]].head())
print("Total rows:", len(df))
""")

md("### Class distribution")
code(r"""
counts = df["label"].value_counts().sort_index()
n_not, n_smile = int(counts.get(0, 0)), int(counts.get(1, 0))
total = n_not + n_smile
print(f"Total       : {total}")
print(f"Not Smiling : {n_not}  ({n_not/total:.1%})")
print(f"Smiling     : {n_smile}  ({n_smile/total:.1%})")
ratio = max(n_not, n_smile) / max(1, min(n_not, n_smile))
print(f"Imbalance   : {ratio:.2f}x  ->", "balanced ✅" if ratio < 1.5 else "will consider class weights")

import matplotlib.pyplot as plt
b = plt.bar(["Not Smiling", "Smiling"], [n_not, n_smile], color=["#94a3b8", "#f59e0b"])
plt.bar_label(b, fmt="%d"); plt.title("CelebA — smile distribution"); plt.ylabel("images")
plt.tight_layout(); plt.show()
""")

md("### Sample images (sanity-check the labels)")
code(r"""
from PIL import Image
def show_samples(label, title, n=8):
    s = df[df.label == label].sample(n, random_state=SEED)
    plt.figure(figsize=(2*n, 2.4))
    for i, (_, r) in enumerate(s.iterrows()):
        plt.subplot(1, n, i+1); plt.imshow(Image.open(r.path).convert("RGB")); plt.axis("off")
    plt.suptitle(title); plt.tight_layout(); plt.show()

show_samples(1, "Smiling (label = 1)")
show_samples(0, "Not Smiling (label = 0)")
""")

md("### Image size check (do we need resizing?)")
code(r"""
sizes = {}
for p in df.path.sample(300, random_state=SEED):
    with Image.open(p) as im:
        sizes[im.size] = sizes.get(im.size, 0) + 1
print("Sizes (width, height) in a 300-image sample:")
for s, c in sorted(sizes.items(), key=lambda kv: -kv[1]):
    print(f"  {s}: {c}")
print("=> we resize everything to 224x224 for MobileNetV2.")
""")

# ── Phase 4: cleaning + balanced subset ──────────────────────────────────────
md(r"""
## Phase 4 — Cleaning + balanced subset
CelebA is curated, but we still (a) verify every chosen image actually opens
(drop corrupt files), and (b) take a **balanced** subset of `PER_CLASS` per class
so Colab training stays fast. This takes ~1–2 minutes.
""")
code(r"""
def verify_image(p):
    try:
        with Image.open(p) as im: im.verify()
        return True
    except Exception:
        return False

# balanced subset: PER_CLASS from each label
parts = []
for lbl, g in df.groupby("label"):
    parts.append(g.sample(min(PER_CLASS, len(g)), random_state=SEED))
sub = pd.concat(parts).reset_index(drop=True)
print("Subset before cleaning:", len(sub))

sub["ok"] = sub["path"].apply(verify_image)          # cleaning pass
bad = int((~sub["ok"]).sum())
sub = sub[sub["ok"]].drop(columns="ok").reset_index(drop=True)
print(f"Dropped {bad} unreadable images. Clean subset: {len(sub)}")
print(sub["label"].value_counts().rename({0: "not_smiling", 1: "smiling"}))
""")

# ── Phase 5: split ───────────────────────────────────────────────────────────
md("## Phase 5 — Train / Validation / Test split (70 / 15 / 15, stratified)")
code(r"""
from sklearn.model_selection import train_test_split
train_df, temp_df = train_test_split(sub, test_size=0.30, stratify=sub.label, random_state=SEED)
val_df,  test_df  = train_test_split(temp_df, test_size=0.50, stratify=temp_df.label, random_state=SEED)
for name, d in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
    print(f"{name:5s}: {len(d):6d}  (smiling={int(d.label.sum())}, not={int((d.label==0).sum())})")
""")

# ── Phase 6+7: tf.data pipeline ──────────────────────────────────────────────
md(r"""
## Phase 6 + 7 — Preprocessing pipeline (resize) + data pipeline
We build fast `tf.data` pipelines straight from file paths (no copying files).
Each image is decoded as RGB and resized to 224×224. **Scaling to [-1, 1]
(MobileNetV2's `preprocess_input`) is baked INTO the model** in the next step —
so the backend later only needs to feed raw 0–255 RGB images. Augmentation is
also part of the model (active only during training).
""")
code(r"""
def make_ds(dframe, training):
    paths  = dframe["path"].values
    labels = dframe["label"].values.astype("float32")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(dframe), seed=SEED, reshuffle_each_iteration=True)
    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=3)          # force 3-channel RGB
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])  # -> float32 in [0,255]
        return img, label
    return ds.map(load, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = make_ds(train_df, training=True)
val_ds   = make_ds(val_df,   training=False)
test_ds  = make_ds(test_df,  training=False)
print("Pipelines ready:", train_ds.element_spec)
""")

# ── Phase 8: model ───────────────────────────────────────────────────────────
md(r"""
## Phase 8 — MobileNetV2 transfer-learning model
Frozen ImageNet-pretrained MobileNetV2 backbone + a small classifier head.
Augmentation and `preprocess_input` live inside the model, so it's self-contained.
""")
code(r"""
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),      # small — keep smile features intact
    layers.RandomZoom(0.10),
], name="augmentation")

base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         include_top=False, weights="imagenet")
base_model.trainable = False          # freeze for the initial phase

inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))   # raw RGB 0-255
x = data_augmentation(inputs)
x = preprocess_input(x)                                # scale to [-1, 1]
x = base_model(x, training=False)                      # keep BatchNorm in inference mode
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)     # P(smiling)
model = keras.Model(inputs, outputs, name="smile_classifier")

model.compile(optimizer=keras.optimizers.Adam(INITIAL_LR),
              loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
""")

# ── Phase 9: initial training ────────────────────────────────────────────────
md("## Phase 9 — Initial training (base frozen, train only the head)")
code(r"""
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
]
history = model.fit(train_ds, validation_data=val_ds,
                    epochs=INITIAL_EPOCHS, callbacks=callbacks)
""")

# ── Phase 10: fine-tuning ────────────────────────────────────────────────────
md(r"""
## Phase 10 — Fine-tuning (unfreeze the top of MobileNetV2)
We unfreeze the later layers and continue training with a **very small** learning
rate so the pretrained features adapt to smiles without being destroyed.
""")
code(r"""
base_model.trainable = True
for layer in base_model.layers[:FINETUNE_AT]:     # keep early layers frozen
    layer.trainable = False
trainable = sum(l.trainable for l in base_model.layers)
print(f"Trainable base layers: {trainable}/{len(base_model.layers)}")

model.compile(optimizer=keras.optimizers.Adam(FINETUNE_LR),
              loss="binary_crossentropy", metrics=["accuracy"])

total_epochs = INITIAL_EPOCHS + FINETUNE_EPOCHS
history_ft = model.fit(train_ds, validation_data=val_ds,
                       epochs=total_epochs, initial_epoch=len(history.epoch),
                       callbacks=callbacks)
""")

md("### Training curves")
code(r"""
acc      = history.history["accuracy"]     + history_ft.history["accuracy"]
val_acc  = history.history["val_accuracy"] + history_ft.history["val_accuracy"]
loss     = history.history["loss"]         + history_ft.history["loss"]
val_loss = history.history["val_loss"]     + history_ft.history["val_loss"]
split_at = len(history.history["accuracy"])

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label="train"); plt.plot(val_acc, label="val")
plt.axvline(split_at-0.5, ls="--", c="gray"); plt.title("Accuracy"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss, label="train"); plt.plot(val_loss, label="val")
plt.axvline(split_at-0.5, ls="--", c="gray"); plt.title("Loss (dashed = fine-tune start)"); plt.legend()
plt.tight_layout(); plt.show()
""")

# ── Phase 11: evaluation ─────────────────────────────────────────────────────
md("## Phase 11 — Evaluation on the held-out test set")
code(r"""
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test accuracy: {test_acc:.4f}   Test loss: {test_loss:.4f}\n")

y_true = np.concatenate([y.numpy() for _, y in test_ds]).astype(int)  # test_ds is not shuffled
y_prob = model.predict(test_ds, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (test set)"); plt.show()
""")

md("### See some predictions")
code(r"""
images, labels = next(iter(test_ds))
probs = model.predict(images, verbose=0).ravel()
plt.figure(figsize=(14, 7))
for i in range(min(12, len(images))):
    plt.subplot(3, 4, i+1)
    plt.imshow(images[i].numpy().astype("uint8")); plt.axis("off")
    p = probs[i]; pred = "Smiling" if p >= 0.5 else "Not Smiling"; conf = p if p >= 0.5 else 1-p
    true = "Smiling" if labels[i].numpy() >= 0.5 else "Not Smiling"
    plt.title(f"{pred} {conf:.0%}\n(true: {true})",
              color="green" if pred == true else "red", fontsize=9)
plt.tight_layout(); plt.show()
""")

# ── Phase 12: save + download ────────────────────────────────────────────────
md(r"""
## Phase 12 — Save the model and download it
Saved in the native Keras format (`.keras`). Put the downloaded file at
`backend/model/smile_classifier.keras` in your project.
""")
code(r"""
model.save(MODEL_OUT)
print("Saved:", MODEL_OUT, "-", round(os.path.getsize(MODEL_OUT)/1e6, 1), "MB")

# Download to your laptop:
try:
    from google.colab import files
    files.download(MODEL_OUT)
except Exception:
    print("Download didn't auto-start — grab it from the Files panel on the left (📁).")

# Alternative — save to Google Drive so it survives session end:
# from google.colab import drive; drive.mount('/content/drive')
# model.save('/content/drive/MyDrive/smile_classifier.keras')
""")

md(r"""
## ✅ Done — training complete!
**Next:** put `smile_classifier.keras` into `backend/model/`, then we build the
FastAPI backend (OpenCV face detection + `/predict`) and the React frontend on your laptop.

Reminder — the model expects **raw 0–255 RGB images resized to 224×224**;
preprocessing is baked in, so the backend must NOT scale pixels again.
""")

# ─────────────────────────────────────────────────────────────────────────────
nb = nbf.v4.new_notebook()
nb.cells = C
nb.metadata.update({
    "kernelspec":   {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "accelerator":  "GPU",
    "colab":        {"provenance": [], "toc_visible": True},
})
nbf.validate(nb)
out = "smile_classifier_training.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote {out} with {len(C)} cells.")
