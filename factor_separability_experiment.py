"""
Factor Separability Experiment
=============================
CNLSA Reframe: VAE-Induced Semantic Drift = Loss of Factor Separability

H₀: corr(F_s^orig, F_p^orig) = corr(F_s^recon, F_p^recon) — VAE preserves factor independence
H₁: corr(F_s^recon, F_p^recon) > corr(F_s^orig, F_p^orig) — VAE increases entanglement

Confirm condition: delta > 0 with bootstrap p < 0.05 AND CI lower bound > 0
Falsify condition: delta ≤ 0 OR p ≥ 0.05 OR CI includes 0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import silhouette_score
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
import torchvision.transforms as T
import torchvision.datasets as datasets
from PIL import Image
import timm
import json
import time
import os
import warnings
warnings.filterwarnings('ignore')

# ============ CONFIG ============
N_IMAGES = 50
BATCH_SIZE = 5
VAE_LATENT_DIM = 256
N_BOOTSTRAP = 1000
RANDOM_SEED = 42
EPOCHS = 5
AUTOENCODER_TRAIN_SAMPLES = 1000
DELTA_THRESHOLD = 0.0

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = 'cpu'
print(f"[CONFIG] Device: {DEVICE}")
print(f"[CONFIG] n_images={N_IMAGES}, n_bootstrap={N_BOOTSTRAP}, epochs={EPOCHS}")

timing = {}

# ============ 1. LOAD MODELS ============
print("\n[1/8] Loading models (via timm)...")
t0 = time.time()

# CLIP (semantic factor) via timm - 768-dim
clip_model = timm.create_model('vit_base_patch32_clip_224', pretrained=True, num_classes=0, img_size=224)
clip_model = clip_model.to(DEVICE)
clip_model.eval()
CLIP_DIM = clip_model.num_features  # 768

# DINOv2 (structural factor) via timm - 384-dim (ViT-S/14)
dino_model = timm.create_model('vit_small_patch14_dinov2', pretrained=True, num_classes=0, img_size=224)
dino_model = dino_model.to(DEVICE)
dino_model.eval()
DINO_DIM = dino_model.num_features  # 384

print(f"[MODEL] CLIP dim: {CLIP_DIM}, DINOv2 dim: {DINO_DIM}")

timing['model_loading'] = time.time() - t0
print(f"[TIME] Model loading: {timing['model_loading']:.1f}s")

# ============ 2. DATASET ============
print("\n[2/8] Preparing dataset...")
t0 = time.time()

cifar_data = datasets.CIFAR100(root='/tmp/cifar100', train=True, download=True)

# Select N_IMAGES diverse samples from different categories
all_labels = np.array(cifar_data.targets)
unique_labels = np.unique(all_labels)
np.random.shuffle(unique_labels)
selected_labels = unique_labels[:N_IMAGES]

label_to_idx = {label: [] for label in selected_labels}
for idx, label in enumerate(all_labels):
    if label in label_to_idx and len(label_to_idx[label]) < 3:
        label_to_idx[label].append(idx)
    if all(len(v) >= 1 for v in label_to_idx.values()):
        break

images = []
category_labels = []
for label in selected_labels:
    indices = label_to_idx[label]
    if indices:
        img_idx = np.random.choice(indices)
        img_arr = cifar_data.data[img_idx]
        images.append(img_arr)
        category_labels.append(label)
    if len(images) >= N_IMAGES:
        break

images = images[:N_IMAGES]
category_labels = category_labels[:N_IMAGES]
n_actual = len(images)
print(f"[DATA] Loaded {n_actual} images from CIFAR-100")

timing['dataset'] = time.time() - t0
print(f"[TIME] Dataset preparation: {timing['dataset']:.1f}s")

# ============ 3. SIMPLE CNN AUTOENCODER ============
print("\n[3/8] Building Simple CNN Autoencoder...")
t0 = time.time()

class SimpleAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, latent_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

autoencoder = SimpleAutoencoder(in_channels=3, latent_dim=VAE_LATENT_DIM)
autoencoder = autoencoder.to(DEVICE)
autoencoder.eval()

timing['autoencoder_build'] = time.time() - t0
print(f"[TIME] Autoencoder build: {timing['autoencoder_build']:.1f}s")

# ============ 4. TRAIN AUTOENCODER ============
print("\n[4/8] Training autoencoder...")
t0 = time.time()

class SubsetDataset(Dataset):
    def __init__(self, data, n_max, transform=None):
        self.data = data
        self.n_max = n_max
        self.transform = transform
        self.flat = isinstance(data[0], np.ndarray)

    def __len__(self):
        return min(self.n_max, len(self.data))

    def __getitem__(self, idx):
        img = self.data[idx]
        if self.flat:
            img = Image.fromarray(img.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img

autoencoder_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

train_subset = SubsetDataset(cifar_data.data, n_max=AUTOENCODER_TRAIN_SAMPLES, transform=autoencoder_transform)
train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)

optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()

autoencoder.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        recon = autoencoder(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 2 == 0:
        print(f"  [AE Epoch {epoch+1}/{EPOCHS}] loss: {epoch_loss/len(train_loader):.4f}")

autoencoder.eval()
timing['autoencoder_train'] = time.time() - t0
print(f"[TIME] Autoencoder training: {timing['autoencoder_train']:.1f}s")

# ============ 5. FEATURE EXTRACTION ============
print("\n[5/8] Extracting features...")
t0 = time.time()

dino_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

def extract_clip_features(images_arrays, batch_size=BATCH_SIZE):
    features = []
    for i in range(0, len(images_arrays), batch_size):
        batch_arr = images_arrays[i:i+batch_size]
        batch_t = torch.stack([autoencoder_transform(Image.fromarray(arr.astype(np.uint8))) for arr in batch_arr]).to(DEVICE)
        with torch.no_grad():
            feats = clip_model(batch_t)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        features.append(feats.cpu().numpy())
    return np.concatenate(features, axis=0)

def extract_dino_features(images_arrays, batch_size=BATCH_SIZE):
    features = []
    for i in range(0, len(images_arrays), batch_size):
        batch_arr = images_arrays[i:i+batch_size]
        batch_t = torch.stack([dino_transform(Image.fromarray(arr.astype(np.uint8))) for arr in batch_arr]).to(DEVICE)
        with torch.no_grad():
            feats = dino_model(batch_t)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        features.append(feats.cpu().numpy())
    return np.concatenate(features, axis=0)

print("  Extracting CLIP features (original)...")
F_s_orig = extract_clip_features(images)
print(f"  CLIP original: {F_s_orig.shape}")

print("  Extracting DINOv2 features (original)...")
F_p_orig = extract_dino_features(images)
print(f"  DINOv2 original: {F_p_orig.shape}")

print("  Running VAE encode-decode...")
recon_arrays = []
for arr in images:
    img_t = autoencoder_transform(Image.fromarray(arr.astype(np.uint8))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        recon_t = autoencoder(img_t)
    recon_np = recon_t.squeeze().permute(1, 2, 0).cpu().numpy()
    recon_np = np.clip(recon_np, 0, 1)
    recon_arrays.append((recon_np * 255).astype(np.uint8))

print("  Extracting CLIP features (reconstructed)...")
F_s_recon = extract_clip_features(recon_arrays)
print(f"  CLIP reconstructed: {F_s_recon.shape}")

print("  Extracting DINOv2 features (reconstructed)...")
F_p_recon = extract_dino_features(recon_arrays)
print(f"  DINOv2 reconstructed: {F_p_recon.shape}")

timing['feature_extraction'] = time.time() - t0
print(f"[TIME] Feature extraction: {timing['feature_extraction']:.1f}s")

# L2 normalize
F_s_orig = F_s_orig / (np.linalg.norm(F_s_orig, axis=1, keepdims=True) + 1e-8)
F_p_orig = F_p_orig / (np.linalg.norm(F_p_orig, axis=1, keepdims=True) + 1e-8)
F_s_recon = F_s_recon / (np.linalg.norm(F_s_recon, axis=1, keepdims=True) + 1e-8)
F_p_recon = F_p_recon / (np.linalg.norm(F_p_recon, axis=1, keepdims=True) + 1e-8)

# ============ 6. METRIC COMPUTATION ============
print("\n[6/8] Computing metrics...")
t0 = time.time()

def mean_pairwise_cosine_cca(F_s, F_p, n_components=1):
    """
    Mean canonical correlation across image-level semantic-structural alignment.
    Uses CCA canonical weight vectors (x_weights_, y_weights_) to project
    CLIP (768-dim) and DINOv2 (384-dim) features into shared canonical space,
    then computes per-image dot product of canonical variates.
    """
    n = len(F_s)
    cca = CCA(n_components=n_components)
    cca.fit(F_s, F_p)

    # Project using canonical weight vectors (avoids sklearn transform X-dim check issue)
    cs_s = F_s.dot(cca.x_weights_)      # (n, n_components)
    cs_p = F_p.dot(cca.y_weights_)      # (n, n_components)

    if n_components == 1:
        scores = cs_s.ravel() * cs_p.ravel()
        return np.mean(scores)
    else:
        scores = np.sum(cs_s * cs_p, axis=1)  # (n,)
        return np.mean(scores)


def mean_pairwise_cosine(F1, F2):
    """Mean of all i<j cosine similarities between F1[i] and F2[j]"""
    n = len(F1)
    sims = []
    for i in range(n):
        for j in range(i+1, n):
            sims.append(np.dot(F1[i], F2[j]))
    return np.mean(sims)

mpcs_orig = mean_pairwise_cosine_cca(F_s_orig, F_p_orig)
mpcs_recon = mean_pairwise_cosine_cca(F_s_recon, F_p_recon)
delta_mpcs = mpcs_recon - mpcs_orig

# CCA canonical correlation (using private scores after fit)
cca = CCA(n_components=1)
cca.fit(F_s_orig, F_p_orig)
r_orig = pearsonr(cca._x_scores.ravel(), cca._y_scores.ravel())[0]

cca.fit(F_s_recon, F_p_recon)
r_recon = pearsonr(cca._x_scores.ravel(), cca._y_scores.ravel())[0]

delta_cca = r_recon - r_orig

# Silhouette
joint_orig = np.hstack([F_s_orig, F_p_orig])
joint_recon = np.hstack([F_s_recon, F_p_recon])

fa_orig = FactorAnalysis(n_components=2, random_state=RANDOM_SEED)
factors_orig = fa_orig.fit_transform(joint_orig)
cluster_labels_orig = (factors_orig[:, 0] > factors_orig[:, 1]).astype(int)
ss_orig = silhouette_score(factors_orig, cluster_labels_orig)

fa_recon = FactorAnalysis(n_components=2, random_state=RANDOM_SEED)
factors_recon = fa_recon.fit_transform(joint_recon)
cluster_labels_recon = (factors_recon[:, 0] > factors_recon[:, 1]).astype(int)
ss_recon = silhouette_score(factors_recon, cluster_labels_recon)

delta_ss = ss_recon - ss_orig

# Linear Probe
le = LabelEncoder()
y = le.fit_transform(category_labels)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(F_p_orig, y)
acc_orig = knn.score(F_p_orig, y)

knn.fit(F_p_recon, y)
acc_recon = knn.score(F_p_recon, y)

delta_acc = acc_recon - acc_orig

timing['metric_computation'] = time.time() - t0
print(f"[TIME] Metric computation: {timing['metric_computation']:.1f}s")

# ============ 7. BOOTSTRAP ============
print(f"\n[7/8] Bootstrap CI (n={N_BOOTSTRAP})...")
t0 = time.time()

def bootstrap_delta(F_s_orig, F_p_orig, F_s_recon, F_p_recon, n_bootstrap=N_BOOTSTRAP):
    deltas = []
    n = len(F_s_orig)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        mpcs_o = mean_pairwise_cosine_cca(F_s_orig[idx], F_p_orig[idx])
        mpcs_r = mean_pairwise_cosine_cca(F_s_recon[idx], F_p_recon[idx])
        deltas.append(mpcs_r - mpcs_o)
    return np.array(deltas)

bootstrap_deltas = bootstrap_delta(F_s_orig, F_p_orig, F_s_recon, F_p_recon)
p_value = float(np.mean(bootstrap_deltas <= 0))
ci_lower = float(np.percentile(bootstrap_deltas, 2.5))
ci_upper = float(np.percentile(bootstrap_deltas, 97.5))

timing['bootstrap'] = time.time() - t0
print(f"[TIME] Bootstrap: {timing['bootstrap']:.1f}s")

# ============ 8. RESULTS ============
print("\n[8/8] Formatting results...")

confirm_mpcs = (delta_mpcs > DELTA_THRESHOLD) and (p_value < 0.05) and (ci_lower > 0)
overall_confirm = confirm_mpcs
overall_falsify = not overall_confirm

confirm_ss = delta_ss < 0
confirm_acc = delta_acc < 0

results = {
    "experiment": "Factor Separability (CNLSA Reframe)",
    "n_images": n_actual,
    "n_bootstrap": N_BOOTSTRAP,
    "models": {
        "clip": "vit_base_patch32_clip_224 (timm, 768-dim)",
        "dino": "vit_small_patch14_dinov2 (timm, 384-dim)",
        "vae": "SimpleAutoencoder CNN"
    },
    "metrics": {
        "mpcs_orig": float(mpcs_orig),
        "mpcs_recon": float(mpcs_recon),
        "delta_mpcs": float(delta_mpcs),
        "r_cca_orig": float(r_orig),
        "r_cca_recon": float(r_recon),
        "delta_cca": float(delta_cca),
        "ss_orig": float(ss_orig),
        "ss_recon": float(ss_recon),
        "delta_ss": float(delta_ss),
        "acc_orig": float(acc_orig),
        "acc_recon": float(acc_recon),
        "delta_acc": float(delta_acc),
    },
    "statistical_tests": {
        "p_value": p_value,
        "ci_95": [ci_lower, ci_upper],
        "delta_threshold": DELTA_THRESHOLD
    },
    "conclusion": {
        "mpcs_result": "CONFIRM" if confirm_mpcs else "FALSIFY",
        "hypothesis": "H₁ supported" if overall_confirm else "H₀ cannot be rejected",
        "p_value": p_value,
        "ci_95": [ci_lower, ci_upper],
        "delta_mpcs": float(delta_mpcs),
        "interpretation": (
            "VAE encode-decode roundtrip INCREASES semantic-structural entanglement. "
            "Factor separability hypothesis CONFIRMED. CLIP and DINOv2 features become "
            "more correlated after VAE roundtrip, consistent with CNLSA mechanism."
            if overall_confirm else
            "VAE encode-decode roundtrip does NOT increase semantic-structural entanglement. "
            "Factor separability hypothesis FALSIFIED for this VAE/feature combination. "
            "The CNLSA semantic drift mechanism operates through channels other than factor entanglement."
        )
    },
    "timing_seconds": {k: float(v) for k, v in timing.items()},
    "random_seed": RANDOM_SEED
}

output_dir = "/home/kas/.openclaw/workspace-domain/research/0418-nova-factor-separability"
os.makedirs(output_dir, exist_ok=True)
results_path = os.path.join(output_dir, "results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n" + "=" * 65)
print("FACTOR SEPARABILITY EXPERIMENT RESULTS")
print("=" * 65)
print(f"MPCS (mean pairwise cosine similarity):")
print(f"  Original:     {mpcs_orig:.4f}")
print(f"  Reconstructed:{mpcs_recon:.4f}")
print(f"  Δ_MPCS:       {delta_mpcs:+.4f}")
print(f"\nCCA canonical r:")
print(f"  Original:     {r_orig:.4f}")
print(f"  Reconstructed:{r_recon:.4f}")
print(f"  Δ_CCA:        {delta_cca:+.4f}")
print(f"\nSilhouette Score (higher = better separability):")
print(f"  Original:     {ss_orig:.4f}")
print(f"  Reconstructed:{ss_recon:.4f}")
print(f"  Δ_Silhouette: {delta_ss:+.4f}  [{'CONFIRM' if confirm_ss else 'no confirm'}]")
print(f"\nLinear Probe Accuracy (structural → semantic):")
print(f"  Original:     {acc_orig:.4f}")
print(f"  Reconstructed:{acc_recon:.4f}")
print(f"  Δ_Acc:        {delta_acc:+.4f}  [{'CONFIRM' if confirm_acc else 'no confirm'}]")
print(f"\nBootstrap (n={N_BOOTSTRAP}):")
print(f"  p-value:      {p_value:.4f}")
print(f"  95% CI:       [{ci_lower:.4f}, {ci_upper:.4f}]")
print("=" * 65)
print(f"\n>>> CONCLUSION: {results['conclusion']['hypothesis']}")
print(f">>> MPCS result: {results['conclusion']['mpcs_result']}")
print(f">>> p-value: {p_value:.4f}, 95% CI [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f">>> Δ_MPCS = {delta_mpcs:+.4f}")
if overall_confirm:
    print("\n>>> H₁ CONFIRMED: VAE roundtrip increases semantic-structural entanglement.")
    print("    CNLSA factor separability mechanism IS supported by this data.")
else:
    print("\n>>> H₀ NOT REJECTED: VAE roundtrip does NOT increase semantic-structural entanglement.")
    print("    CNLSA factor separability mechanism NOT supported by this data.")
print("=" * 65)

total_time = sum(timing.values())
print(f"\n[TIMING TOTAL] {total_time:.1f}s ({total_time/60:.1f} min)")
for k, v in timing.items():
    print(f"  {k}: {v:.1f}s")
print(f"\n[OUTPUT] results.json → {results_path}")
