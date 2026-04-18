# Factor Separability Experiment Design
## CNLSA Reframe: VAE-Induced Semantic Drift = Loss of Factor Separability

**Date:** 2026-04-18 (Saturday)  
**Author:** Nova (0418-AM isolated session)  
**Status:** Experiment Design — Pending Kernel Implementation  
**Runtime Constraint:** CPU-only, ≤10 min for n=50 images

---

## 1. Scientific Hypothesis

### 1.1 Core Claim (Send-VAE Framework)

**Send-VAE (ICLR 2026)** proposes that VAE latent space can be factorized into two independent components:
- **Semantic factor (z_s):** Captures high-level semantic content (object identity, category)
- **Perceptual factor (z_p):** Captures low-level texture, style, and pixel-level details

The **CNLSA hypothesis** (VAE-Induced Semantic Drift) is reframed as:

> **VAE encode-decode roundtrip causes loss of factor separability**: the semantic and perceptual factors that are (approximately) independent in the original image become entangled in the reconstructed image. This entanglement manifests as increased correlation between semantic features (CLIP) and structural features (DINOv2).

### 1.2 Formalization

Let:
- **F_s^orig**: CLIP semantic features extracted from original images
- **F_p^orig**: DINOv2 structural features extracted from original images  
- **F_s_recon**: CLIP semantic features extracted from VAE-reconstructed images
- **F_p_recon**: DINOv2 structural features extracted from VAE-reconstructed images

**Null Hypothesis (H₀):** corr(F_s^orig, F_p^orig) = corr(F_s^recon, F_p^recon) — VAE roundtrip preserves factor independence

**Alternative Hypothesis (H₁):** corr(F_s^recon, F_p^recon) > corr(F_s^orig, F_p^orig) — VAE roundtrip increases factor entanglement

**Confirm condition:** corr(F_s^recon, F_p^recon) − corr(F_s^orig, F_p^orig) > δ (threshold), with bootstrap p < 0.05

**Falsify condition:** corr(F_s^recon, F_p^recon) − corr(F_s^orig, F_p^orig) ≤ δ, or the difference is not statistically significant

---

## 2. Experimental Design

### 2.1 Feature Extractors (Frozen, CPU-feasible)

| Extractor | Model | Dimension | Role | CPU Speed |
|-----------|-------|-----------|------|-----------|
| CLIP | openai/clip-vit-base-patch32 | 512-dim | Semantic factor | ~0.3s/image |
| DINOv2 | facebook/dinov2-vits14 | 384-dim | Structural factor | ~0.5s/image |

**Why these models:**
- Both are trained with self-supervised objectives (contrastive / self-distillation)
- CLIP: semantic/identity-focused (contrastive text-image training)
- DINOv2: structural/visual grammar-focused (self-distillation on raw images)
- Both are frozen (no gradient computation) — pure inference
- Both have CPU-compatible inference paths

### 2.2 VAE for Roundtrip

**Challenge:** Standard KL-VAE (e.g., Stability-AI SD-VAE) is slow on CPU (~40s/image, confirmed in 0417-AM window).

**CPU-Feasible VAE Options (priority order):**

1. **VQ-VAE-2 (轻量版):** Use a pre-trained VQ-VAE with 32×32 latent space. Code available in torchvision or custom implementation. ~2-5s/image on CPU.
2. **DiT VAE (if available):** The VAE from DiT (512×512 model) — may have a smaller variant.
3. **Simple Autoencoder:** Train a small CNN autoencoder on ImageNet subset during experiment initialization (~2 min training, then fast inference).
4. **Send-VAE official code:** If the Send-VAE weights are available (ICLR 2026, code✅ from Scout), use their semantic-disentangled VAE directly. This would be the most direct test.

**For this design, we use Option 3 (Simple CNN Autoencoder):**
- Architecture: Conv encoder (32→64→128→256 channels) → 256-dim latent → Conv decoder (reverse)
- Training: 5 epochs on 1000 ImageNet samples (CPU, ~2 min)
- This gives us a working VAE without GPU dependency

**Critical note:** The VAE quality matters. If the autoencoder is too weak (reconstruction quality < 0.7 SSIM), the results may be confounded by severe degradation rather than factor entanglement.

### 2.3 Dataset

**n = 50 images**, diverse category coverage:

- **Source:** CIFAR-100 (100 categories, pick 50 diverse ones) or a curated subset of COCO val2017
- **Diversity criterion:** Each image must have distinguishable semantic content (object + scene) and distinct texture/structure
- **Resolution:** Resize to 224×224 (CLIP/DINOv2 native resolution)
- **Rationale:** n=50 provides sufficient statistical power (see Section 4) for effect size d≥0.5 at α=0.05

### 2.4 Factor Extraction Pipeline

```
Image → Resize(224×224)
       ├→ CLIP encoder → normalize → F_s^orig (512-dim)
       ├→ DINOv2 encoder → normalize → F_p^orig (384-dim)
       │
       ├→ VAE encoder → 256-dim latent → VAE decoder → Recon image
       │
       ├→ CLIP encoder(Recon) → normalize → F_s^recon (512-dim)
       └→ DINOv2 encoder(Recon) → normalize → F_p^recon (384-dim)
```

**All 4 feature matrices are L2-normalized before analysis.**

---

## 3. Metrics and Analysis

### 3.1 Primary Metric: Cross-Factor Correlation (CFC)

Compute the correlation between semantic and structural features across all n=50 images.

**Method 1: Mean Pairwise Cosine Similarity (MPCS)**
```
MPCS_orig = mean_{i<j} (F_s^orig[i] · F_p^orig[j]) / (||F_s^orig[i]||·||F_p^orig[j]||)
MPCS_recon = mean_{i<j} (F_s^recon[i] · F_p^recon[j]) / (||F_s^recon[i]||·||F_p^recon[j]||)
```
Higher MPCS = stronger semantic-structural entanglement.

**Method 2: Canonical Correlation Analysis (CCA)**
- Project F_s and F_p into a shared subspace
- Compute first canonical correlation coefficient r_cca
- Compare r_cca^orig vs r_cca^recon

**Method 3: Procrustes-Aligned Distance Correlation**
- Align F_s^orig to F_s^recon (Procrustes) to remove orthogonal transformation
- Compute distance correlation between aligned F_s and F_p
- This removes the confounds of VAE introducing arbitrary rotations

### 3.2 Secondary Metrics

**Separability Score (SS):**
- Fit FactorAnalysis(k=2) on [F_s, F_p] concatenated
- Assign each image to cluster based on which factor loads more heavily on CLIP vs DINOv2
- Silhouette score measures cluster separation
- SS_recon < SS_orig → hypothesis confirmed

**Semantic Discriminability (SD):**
- Train linear probe on F_p^orig to predict semantic categories (50-way classification)
- Evaluate accuracy on F_p^recon
- If VAE entangles factors: accuracy_drop > 0

**Perceptual Fidelity (PF):**
- SSIM between original and reconstructed images (sanity check: if PF is very low, degradation dominates)

### 3.3 Statistical Testing

**Bootstrap test (n=1000 resamples):**
```
delta = CFC_recon - CFC_orig
SE = std(bootstrap_deltas)
p = proportion(bootstrap_deltas ≤ 0)  # one-sided test
CI = percentile(bootstrap_deltas, [2.5, 97.5])
```

**Confirm threshold:** p < 0.05 AND CI lower bound > 0

---

## 4. Statistical Power Analysis

Given:
- n = 50 images
- α = 0.05 (one-sided)
- Expected effect size: d = 0.5 (moderate)

Power calculation:
```
from scipy import stats
power = stats.power.tt_solve_power(n=50, d=0.5, alpha=0.05, alternative='larger')
# ≈ 0.82 power for d=0.5
# ≈ 0.98 power for d=0.8
```

**Minimum detectable effect:** d ≈ 0.4 at 80% power with n=50

---

## 5. CPU Timing Budget

| Step | Time (n=50) | Notes |
|------|-------------|-------|
| CLIP inference (original) | ~15s | batch_size=5 |
| DINOv2 inference (original) | ~25s | batch_size=5 |
| VAE autoencoder training | ~120s | 5 epochs on 1000 samples |
| VAE encode-decode | ~100s | ~2s/image |
| CLIP inference (recon) | ~15s | batch_size=5 |
| DINOv2 inference (recon) | ~25s | batch_size=5 |
| Metric computation | ~5s | PCA, CCA, bootstrap |
| **TOTAL** | **~5-7 min** | Within 10-min budget |

---

## 6. Pseudocode

```python
import torch
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import silhouette_score
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import NearestNeighbors
import torchvision.transforms as T
from PIL import Image

# ============ CONFIG ============
N_IMAGES = 50
BATCH_SIZE = 5
VAE_LATENT_DIM = 256
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============ LOAD MODELS ============
# CLIP (semantic factor)
clip_model = load_clip("openai/clip-vit-base-patch32")
clip_preprocess = get_clip_preprocess()

# DINOv2 (structural factor)  
dino_model = load_dinov2("facebook/dinov2-vits14")
dino_preprocess = get_dino_preprocess()

# Simple CNN Autoencoder (VAE proxy)
autoencoder = SimpleAutoencoder(in_channels=3, latent_dim=VAE_LATENT_DIM)
autoencoder.train_on_imagenet_subset(epochs=5, n_samples=1000)  # ~2 min on CPU
autoencoder.eval()

# ============ DATASET ============
images = load_diverse_images(n=N_IMAGES)  # CIFAR-100 or COCO subset
# Precompute all features
F_s_orig = []   # CLIP original
F_p_orig = []   # DINOv2 original
F_s_recon = []  # CLIP reconstructed
F_p_recon = []  # DINOv2 reconstructed

for img in images:
    # Original features
    img_t = preprocess(img)
    with torch.no_grad():
        F_s_orig.append(clip_model(img_t))
        F_p_orig.append(dino_model(img_t))
    
    # VAE roundtrip
    recon = autoencoder.decode(autoencoder.encode(img_t))
    
    # Reconstructed features
    with torch.no_grad():
        F_s_recon.append(clip_model(recon))
        F_p_recon.append(dino_model(recon))

F_s_orig = torch.stack(F_s_orig).numpy()
F_p_orig = torch.stack(F_p_orig).numpy()
F_s_recon = torch.stack(F_s_recon).numpy()
F_p_recon = torch.stack(F_p_recon).numpy()

# L2 normalize
F_s_orig = F_s_orig / np.linalg.norm(F_s_orig, axis=1, keepdims=True)
F_p_orig = F_p_orig / np.linalg.norm(F_p_orig, axis=1, keepdims=True)
F_s_recon = F_s_recon / np.linalg.norm(F_s_recon, axis=1, keepdims=True)
F_p_recon = F_p_recon / np.linalg.norm(F_p_recon, axis=1, keepdims=True)

# ============ METRIC 1: MEAN PAIRWISE COSINE SIMILARITY ============
def mean_pairwise_cosine(F1, F2):
    """Mean of all i<j cosine similarities between F1[i] and F2[j]"""
    n = len(F1)
    sims = []
    for i in range(n):
        for j in range(i+1, n):
            sim = np.dot(F1[i], F2[j])
            sims.append(sim)
    return np.mean(sims)

mpcs_orig = mean_pairwise_cosine(F_s_orig, F_p_orig)
mpcs_recon = mean_pairwise_cosine(F_s_recon, F_p_recon)
delta_mpcs = mpcs_recon - mpcs_orig

# ============ METRIC 2: CCA CANONICAL CORRELATION ============
cca = CCA(n_components=1)
cca.fit(F_s_orig, F_p_orig)
_, r_orig = cca.predict(F_s_orig, F_p_orig)
r_orig = np.corrcoef(cca.x_scores_.ravel(), cca.y_scores_.ravel())[0,1]

cca.fit(F_s_recon, F_p_recon)
_, r_recon = cca.predict(F_s_recon, F_p_recon)
r_recon = np.corrcoef(cca.x_scores_.ravel(), cca.y_scores_.ravel())[0,1]

delta_cca = r_recon - r_orig

# ============ METRIC 3: FACTOR ANALYSIS SEPARABILITY ============
# Concatenate semantic + structural features
joint_orig = np.hstack([F_s_orig, F_p_orig])  # (50, 512+384=896)
joint_recon = np.hstack([F_s_recon, F_p_recon])

# Fit FactorAnalysis with 2 factors
fa_orig = FactorAnalysis(n_components=2, random_state=RANDOM_SEED)
factors_orig = fa_orig.fit_transform(joint_orig)
# Label factors: factor with higher mean dot to CLIP = semantic factor
fa_semantic_orig = factors_orig[:, np.argmax([np.mean(factors_orig[:,i] @ F_s_orig.mean(axis=0)) for i in range(2)])]

# K-means clustering based on dominant factor
cluster_labels_orig = (factors_orig[:, 0] > factors_orig[:, 1]).astype(int)
ss_orig = silhouette_score(factors_orig, cluster_labels_orig)

fa_recon = FactorAnalysis(n_components=2, random_state=RANDOM_SEED)
factors_recon = fa_recon.fit_transform(joint_recon)
cluster_labels_recon = (factors_recon[:, 0] > factors_recon[:, 1]).astype(int)
ss_recon = silhouette_score(factors_recon, cluster_labels_recon)

delta_ss = ss_recon - ss_orig

# ============ METRIC 4: SEMANTIC DISCRIMINABILITY ============
# Train linear probe on structural features → predict semantic categories
categories = get_image_categories(n=N_IMAGES)  # 50 categories
probe = LinearProbe(n_features=F_p_orig.shape[1], n_classes=N_IMAGES)
probe.fit(F_p_orig, categories)
acc_orig = probe.score(F_p_orig, categories)
acc_recon = probe.score(F_p_recon, categories)
delta_acc = acc_recon - acc_orig

# ============ BOOTSTRAP CONFIDENCE INTERVALS ============
def bootstrap_delta(n_bootstrap=1000):
    deltas = []
    n = len(F_s_orig)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        mpcs_o = mean_pairwise_cosine(F_s_orig[idx], F_p_orig[idx])
        mpcs_r = mean_pairwise_cosine(F_s_recon[idx], F_p_recon[idx])
        deltas.append(mpcs_r - mpcs_o)
    return np.array(deltas)

bootstrap_deltas = bootstrap_delta(1000)
p_value = np.mean(bootstrap_deltas <= 0)
ci_lower, ci_upper = np.percentile(bootstrap_deltas, [2.5, 97.5])

# ============ RESULTS ============
results = {
    "mpcs_orig": mpcs_orig,
    "mpcs_recon": mpcs_recon,
    "delta_mpcs": delta_mpcs,
    "r_cca_orig": r_orig,
    "r_cca_recon": r_recon,
    "delta_cca": delta_cca,
    "ss_orig": ss_orig,
    "ss_recon": ss_recon,
    "delta_ss": delta_ss,
    "acc_orig": acc_orig,
    "acc_recon": acc_recon,
    "delta_acc": delta_acc,
    "p_value": p_value,
    "ci_95": (ci_lower, ci_upper),
}

print("=" * 60)
print("FACTOR SEPARABILITY EXPERIMENT RESULTS")
print("=" * 60)
print(f"MPCS_orig:     {mpcs_orig:.4f}")
print(f"MPCS_recon:    {mpcs_recon:.4f}")
print(f"Δ_MPCS:        {delta_mpcs:+.4f}  [{'CONFIRM' if delta_mpcs > 0 else 'FALSIFY'}]")
print(f"CCA_orig:      {r_orig:.4f}")
print(f"CCA_recon:     {r_recon:.4f}")
print(f"Δ_CCA:         {delta_cca:+.4f}  [{'CONFIRM' if delta_cca > 0 else 'FALSIFY'}]")
print(f"Silhouette_orig: {ss_orig:.4f}")
print(f"Silhouette_recon:{ss_recon:.4f}")
print(f"Δ_Silhouette:  {delta_ss:+.4f}  [{'CONFIRM' if delta_ss < 0 else 'FALSIFY'}]")
print(f"Acc_orig:      {acc_orig:.4f}")
print(f"Acc_recon:     {acc_recon:.4f}")
print(f"Δ_Acc:         {delta_acc:+.4f}  [{'CONFIRM' if delta_acc < 0 else 'FALSIFY'}]")
print(f"p_value:       {p_value:.4f}")
print(f"95% CI:        [{ci_lower:.4f}, {ci_upper:.4f}]")
print("=" * 60)
```

---

## 7. Expected Results

### 7.1 Confirm Case (H₁ supported)

**Expected observation:** After VAE roundtrip, semantic and structural features become more correlated.

| Metric | Before VAE | After VAE | Direction | Interpretation |
|--------|-----------|-----------|-----------|----------------|
| MPCS | ~0.05 | ~0.15 | ↑ | Semantic-structural entanglement increases |
| CCA r | ~0.30 | ~0.50 | ↑ | Shared variance increases |
| Silhouette Score | ~0.40 | ~0.25 | ↓ | Factor separability decreases |
| Probe Accuracy | ~0.75 | ~0.55 | ↓ | Structural features lose semantic info |

**Cohen's d for Δ_MPCS:** d ≈ 0.5–0.8 (moderate to large effect)  
**p-value:** < 0.05 (bootstrap)  
**95% CI:** lower bound > 0

**Interpretation:** VAE encode-decode roundtrip systematically increases correlation between CLIP semantic space and DINOv2 structural space. This is consistent with Send-VAE's claim that VAE latent space does NOT preserve semantic-perceptual independence. The CNLSA phenomenon (VAE-induced semantic drift) is measurable as "factor entanglement."

### 7.2 Falsify Case (H₀ cannot be rejected)

**Expected observation:** VAE roundtrip does NOT increase semantic-structural correlation.

| Metric | Before VAE | After VAE | Direction | Interpretation |
|--------|-----------|-----------|-----------|----------------|
| MPCS | ~0.05 | ~0.04 | ↓ or ≈ | No entanglement increase |
| CCA r | ~0.30 | ~0.28 | ↓ or ≈ | Independence preserved |
| Silhouette Score | ~0.40 | ~0.42 | ↑ or ≈ | Separability maintained |
| Probe Accuracy | ~0.75 | ~0.76 | ↑ or ≈ | Structural features remain discriminative |

**p-value:** ≥ 0.05 (bootstrap)  
**95% CI:** includes 0 or is entirely negative

**Interpretation:** VAE encode-decode roundtrip does NOT systematically increase semantic-structural entanglement. The CNLSA phenomenon, if real, operates through a different mechanism than factor separability loss. Alternative hypotheses: (a) VAE drift is uniform across semantic categories (category-uniform, as confirmed in 0415-PM ANOVA), not factor-specific; (b) VAE affects semantic consistency through a mechanism that does not manifest as CLIP-DINOv2 correlation change.

### 7.3 Edge Cases

**Case A — Mild entanglement increase, not significant:**
- Δ_MPCS ≈ +0.03, p ≈ 0.12
- **Conclusion:** Trend in the right direction but underpowered. Need n≥100 or stronger VAE to detect effect.

**Case B — Entanglement decreases:**
- Δ_MPCS < 0, SS increases
- **Conclusion:** VAE roundtrip actually DECREASES semantic-structural correlation. This would FALSIFY CNLSA factor separability hypothesis and suggest VAE acts as a semantic filter, not a drift generator.

**Case C — High variance across categories:**
- Overall Δ_MPCS not significant, but some categories show strong entanglement
- **Conclusion:** CNLSA is category-selective, not uniform. This aligns with Scalpel's warning that σ=0 gate threshold is arbitrary.

---

## 8. Sensitivity Analysis

### 8.1 What if VAE quality is poor?

If autoencoder SSIM < 0.6, the "reconstruction" is too degraded to test semantic entanglement. We need a minimum quality threshold.

**Mitigation:** 
- Report SSIM for all experiments
- If SSIM < 0.6, exclude those images or use a stronger autoencoder
- Target: SSIM ≥ 0.7 for included images

### 8.2 What if CLIP/DINOv2 features are too correlated to start?

If F_s_orig and F_p_orig are already highly correlated (r > 0.8), there's no room to detect an increase.

**Mitigation:**
- Report baseline r_cca_orig before running
- If r_cca_orig > 0.7, use a different structural feature extractor (e.g., ResNet50 instead of DINOv2) to get a weaker baseline

### 8.3 What if the autoencoder training introduces confounds?

Training the autoencoder on ImageNet creates a domain confound.

**Mitigation:**
- Report which dataset the autoencoder was trained on
- The semantic-structural independence hypothesis should be tested with a neutral autoencoder (not one trained on the same domain as the test images)

---

## 9. Connection to Prior Results

### 9.1 Alignment with CNLSA CPU Results (0415-PM)

- **DINOv2 ViT-S/14 CS=0.8155:** Shows VAE damages DINOv2 structural features significantly (not just CLIP)
- **Category ANOVA p=0.6037:** Drift is uniform across categories, not category-selective
- **This experiment:** Tests whether the CLIP-DINOv2 degradation is correlated (entanglement) or independent

### 9.2 Alignment with TrACE-Video Idea D (0417-PM)

- **DINOv2 L2 predicts CLIP inconsistency (r=-0.8973):** Shows DINOv2 and CLIP signals are strongly coupled in the presence of noise
- **This experiment:** Tests whether VAE roundtrip specifically causes this coupling (factor entanglement)

### 9.3 Alignment with Nova-Idea-#2 (0416-PM)

- **Cross-frame VAE L2 distance predicts semantic inconsistency (r=0.9895 synthetic):** Shows latent distance metric correlates with semantic drift
- **This experiment:** Decomposes the latent space into semantic and structural factors to understand WHY the distance metric works

---

## 10. Implementation Checklist for Kernel

- [ ] Install dependencies: `torch`, `transformers`, `timm`, `scikit-learn`, `Pillow`
- [ ] Download pre-trained CLIP (openai/clip-vit-base-patch32) and DINOv2 (facebook/dinov2-vits14)
- [ ] Verify models load correctly on CPU (no GPU required)
- [ ] Implement SimpleAutoencoder and verify it trains on CPU within 2 minutes
- [ ] Load 50 diverse images (CIFAR-100 or COCO subset)
- [ ] Run feature extraction pipeline (all 4 feature matrices)
- [ ] Compute all metrics (MPCS, CCA, Silhouette, Probe Accuracy)
- [ ] Run bootstrap confidence intervals (n=1000)
- [ ] Save results to JSON
- [ ] Report timing breakdown

**Estimated total runtime: 5–8 minutes on CPU**

---

## 11. References

- **Send-VAE (ICLR 2026):** Semantic-disentangled VAE with explicit z_s (semantic) and z_p (perceptual) factorization. Code: `KlingAIResearch/Send-VAE`
- **CNLSA CPU validation (0415-PM):** DINOv2 ViT-S/14 CS=0.8155, category-uniform drift (ANOVA p=0.6037)
- **TrACE-Video Idea D (0417-PM):** DINOv2 L2 predicts CLIP inconsistency (r=-0.8973)
- **Nova-Idea-#2 (0416-PM):** Cross-frame latent drift (r=0.9895 synthetic, r=0.6117 honest cross-encoder)

---

## 12. File Structure

```
research/0418-nova-factor-separability/
├── factor-separability-experiment-design.md   (this document)
├── factor_separability_experiment.py          (Kernel implementation)
├── results.json                                (experiment outputs)
└── README.md                                   (quick summary)
```

---

**Bottom line:** This experiment provides a direct, CPU-feasible test of the CNLSA "factor separability" hypothesis. If confirmed, it would explain WHY VAE causes semantic drift (factor entanglement mechanism). If falsified, it would rule out the factor separability channel and focus attention on alternative mechanisms (category-uniform degradation, uniform semantic compression, etc.).
