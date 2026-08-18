# %% [markdown]
# # AI Smile Classifier — Phase 2: Download & Analyze (CelebA)
#
# Run this in **Google Colab** (Runtime -> Change runtime type -> **GPU**).
#
# HOW TO RUN:
#   - Upload this file to Colab, then copy each "# %%" block into its own cell
#     and run top-to-bottom (recommended, so you can SEE the plots), OR
#   - `!python 01_download_and_analyze.py` (plots won't show inline).
#
# GOAL of this phase: download CelebA, then understand the data BEFORE training:
#   total images, smiling vs not-smiling balance, sample images, image sizes.

# %% [markdown]
# ## Step 0: Install dependencies (run once in Colab)

# %%
# !pip install kagglehub pandas matplotlib pillow -q

# %% [markdown]
# ## Step 1: Kaggle authentication
# You already used kagglehub for your BUSI project, so this may already work.
# If you get an auth error, uncomment ONE of the options below.

# %%
import os

# Option A — interactive login (paste your Kaggle username + API key):
# import kagglehub; kagglehub.login()

# Option B — set credentials directly (from kaggle.json):
# os.environ["KAGGLE_USERNAME"] = "your_username"
# os.environ["KAGGLE_KEY"] = "your_key"
#
# To get the key: kaggle.com -> your avatar -> Settings -> API -> "Create New Token"
# -> downloads kaggle.json which contains {"username": "...", "key": "..."}

# %% [markdown]
# ## Step 2: Download CelebA (~1.4 GB, one-time)

# %%
import kagglehub

DATA_ROOT = kagglehub.dataset_download("jessicali9530/celeba-dataset")
print("Downloaded to:", DATA_ROOT)

# %% [markdown]
# ## Step 3: Locate the files (robust — folder nesting can vary)
# CelebA on Kaggle contains:
#   - img_align_celeba/img_align_celeba/*.jpg   (202,599 aligned face images)
#   - list_attr_celeba.csv                       (40 attributes, incl. "Smiling")

# %%
import glob

# Find the attributes CSV anywhere under the download folder
csv_hits = glob.glob(os.path.join(DATA_ROOT, "**", "list_attr_celeba.csv"), recursive=True)
assert csv_hits, "list_attr_celeba.csv not found — check the download."
ATTR_CSV = csv_hits[0]

# Find the image directory = the folder that holds the most .jpg files
jpg_count_by_dir = {}
for root, _dirs, files in os.walk(DATA_ROOT):
    n = sum(1 for f in files if f.lower().endswith(".jpg"))
    if n:
        jpg_count_by_dir[root] = n
IMG_DIR = max(jpg_count_by_dir, key=jpg_count_by_dir.get)

print("Attributes CSV:", ATTR_CSV)
print("Image dir     :", IMG_DIR)
print("JPGs in image dir:", jpg_count_by_dir[IMG_DIR])

# %% [markdown]
# ## Step 4: Load attributes and build the smile label
# Attribute values are -1 / +1.  We map:  Smiling == 1 -> 1 (smiling),  else -> 0 (not_smiling).

# %%
import pandas as pd

df = pd.read_csv(ATTR_CSV)

# First column is the image filename; normalize its name to "image_id"
if "image_id" not in df.columns:
    df = df.rename(columns={df.columns[0]: "image_id"})

assert "Smiling" in df.columns, f"'Smiling' column missing. Columns: {list(df.columns)[:5]}..."

df["label"] = (df["Smiling"] == 1).astype(int)   # 1 = smiling, 0 = not_smiling
df["path"] = df["image_id"].apply(lambda f: os.path.join(IMG_DIR, f))

print(df[["image_id", "Smiling", "label"]].head())
print("\nTotal rows in CSV:", len(df))

# %% [markdown]
# ## Step 5: Total images + class distribution  (Phase 3 analysis)

# %%
counts = df["label"].value_counts().sort_index()
n_not = int(counts.get(0, 0))
n_smile = int(counts.get(1, 0))
total = n_not + n_smile

print("=== Class distribution ===")
print(f"Total images : {total}")
print(f"Not Smiling  : {n_not}  ({n_not/total:.1%})")
print(f"Smiling      : {n_smile}  ({n_smile/total:.1%})")
ratio = max(n_not, n_smile) / max(1, min(n_not, n_smile))
print(f"Imbalance ratio (majority/minority): {ratio:.2f}x")
print("=> " + ("Well balanced, no special handling needed."
                if ratio < 1.5 else
                "Some imbalance — we'll use class weights when training."))

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(4.5, 3.5))
bars = plt.bar(["Not Smiling", "Smiling"], [n_not, n_smile],
               color=["#94a3b8", "#f59e0b"])
plt.bar_label(bars, fmt="%d")
plt.title("CelebA — Smile class distribution")
plt.ylabel("Number of images")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 6: Look at sample images (sanity check the labels)

# %%
from PIL import Image

def show_samples(label, title, n=8):
    sample = df[df["label"] == label].sample(n, random_state=42)
    plt.figure(figsize=(2 * n, 2.4))
    for i, (_, row) in enumerate(sample.iterrows()):
        plt.subplot(1, n, i + 1)
        plt.imshow(Image.open(row["path"]).convert("RGB"))
        plt.axis("off")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

show_samples(1, "Smiling (label = 1)")
show_samples(0, "Not Smiling (label = 0)")

# %% [markdown]
# ## Step 7: Check image dimensions (do we need resizing?)
# MobileNetV2 wants 224x224x3. CelebA aligned images are all the same size,
# so we'll just resize later. This step confirms there are no surprises.

# %%
sizes = {}
for p in df["path"].sample(300, random_state=42):
    with Image.open(p) as im:
        sizes[im.size] = sizes.get(im.size, 0) + 1  # im.size = (width, height)

print("Unique (width, height) sizes in a 300-image sample:")
for size, cnt in sorted(sizes.items(), key=lambda kv: -kv[1]):
    print(f"  {size}: {cnt} images")

# %% [markdown]
# ## Step 8: Summary
# What you should see:
#   - ~202,599 total images
#   - Smiling ~48%, Not Smiling ~52%  (nicely balanced)
#   - Sample grids clearly show smiling vs neutral faces (labels look correct)
#   - All images one size, e.g. (178, 218)
#
# NEXT (Phase 3): we'll clean, pick a balanced subset (to keep Colab training fast),
# and split 70/15/15 into train/val/test folders.

# %%
print("Phase 2 complete. Report back: total images, the two class counts, and the image size.")
