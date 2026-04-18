# Scout Results 0418-AM: TrACE-Video + CNLSA Paper Survey (60-day window: 2026-02-18 to 2026-04-18)

## Topic A: TrACE-Video / Video Generation Compute Adaptation / Inter-Frame Latent Consistency

### HIGH PRIORITY (Code Available)

#### 1. LIPAR — Training-Free Latent Inter-Frame Pruning with Attention Recovery ⭐⭐⭐⭐⭐
- **arXiv:** 2603.05811 (March 2026)
- **Venue:** arXiv preprint
- **Code:** https://github.com/DennisMenn/lipar
- **Project Page:** https://dennismenn.github.io/lipar/
- **Key Idea:** Detects and skips recomputing duplicated latent patches (temporal/spatial redundancy in latent space). Uses "Attention Recovery" to reduce train-inference mismatch from pruning.
- **Speedup:** 1.45× on Wan2.1 video generation
- **Relevance to TrACE-Video:** DIFFERENT angle (spatial redundancy detection vs semantic consistency metric). But LIPAR validates that latent space has exploitable redundancy - supports the premise that VAE latents contain semantic structure. TrACE-Video could use LCS score as the gating signal for LIPAR-style pruning decisions.
- **Relevance to CNLSA:** HIGH - LIPAR's core finding: latent patches have temporal redundancy that can be detected without generation. This supports CNLSA's premise that VAE latents encode semantic information.

#### 2. Pathwise TTC — Pathwise Test-Time Correction for Autoregressive Long Video Generation ⭐⭐⭐⭐
- **arXiv:** 2602.05871 (February 2026, v2 March 2026)
- **Venue:** arXiv preprint
- **Code:** https://github.com/xbxsxp9/Pathwise_TTC
- **Key Idea:** Uses initial frame as stable reference anchor to calibrate intermediate stochastic states along the diffusion sampling trajectory. Training-free, "negligible overhead."
- **Addresses:** Error accumulation in distilled AR video generation for long sequences (30-second benchmarks).
- **Relevance to TrACE-Video:** VERY HIGH - Directly related to test-time intervention for temporal consistency. If TrACE-Video LCS predicts drift, TTC could be the CORRECTION mechanism triggered by low LCS. Pipeline: TrACE-Video (detect) → TTC (correct).
- **Relevance to CNLSA:** MEDIUM - TTC uses initial frame as anchor, which is a form of semantic reference tracking.

### MEDIUM PRIORITY (Code Available)

#### 3. SFD — Semantic-First Diffusion (CVPR 2026) ⭐⭐⭐⭐
- **arXiv:** 2512.04926
- **Venue:** CVPR 2026
- **Code:** https://github.com/yuemingPAN/SFD
- **Project Page:** https://yuemingpan.github.io/SFD.github.io/
- **Key Idea:** Asynchronous denoising where semantic latents advance ahead of texture latents. Three-phase schedule: semantic initialization → async joint → texture completion.
- **Addresses:** Semantic-texture harmonization in latent diffusion. "Semantics lead the way."
- **Relevance to TrACE-Video:** HIGH - SFD prioritizes semantic structure in latent space. If SFD's semantic latents are more consistent, TrACE-Video's LCS could validate which frames need semantic re-guidance. Complementary: SFD (generation) + TrACE-Video (measurement).
- **Relevance to CNLSA:** HIGH - SFD directly addresses semantic modeling in latent space, suggesting semantic drift is a real phenomenon.

### OTHER RELEVANT (Video Generation / Semantic Consistency)

#### 4. Semantics-Guided Hierarchical Video Prediction (Re2Pix) ⭐⭐⭐
- **arXiv:** 2604.11707 (April 2026)
- **Venue:** arXiv preprint
- **Key Idea:** Hierarchical VFM semantic feature prediction → latent video diffusion. Addresses semantic consistency via multi-level prediction.
- **Relevance:** Related to semantic consistency in video generation.

#### 5. Long-Horizon Streaming Video Generation (Hybrid Forcing) ⭐⭐⭐
- **arXiv:** 2604.10103 (April 2026)
- **Venue:** arXiv preprint
- **Key Idea:** Hybrid local window + linear history aggregation on Wan2.1-T2V-1.3B. 29.5 FPS on H100.
- **Relevance:** Different approach (temporal modeling architecture) but related to long-video consistency.

#### 6. 4D Latent Reward — World-Consistent Video Generation ⭐⭐⭐
- **arXiv:** 2603.26599 (March 2026)
- **Venue:** arXiv preprint
- **Key Idea:** Computes geometry-driven rewards directly in latent space for video post-training.
- **Relevance:** Test-time latent reward computation - related to TrACE-Video's latent space measurement concept.

#### 7. StableWorld — Towards Stable and Consistent Long Interactive Video Generation ⭐⭐
- **arXiv:** 2601.15281 (January 2026)
- **Venue:** arXiv preprint
- **Key Idea:** Stability and temporal consistency in interactive video generation.
- **Relevance:** Directly related to video consistency.

#### 8. TrajLoom — Dense Future Trajectory Generation ⭐⭐
- **arXiv:** 2603.22606 (March 2026)
- **Venue:** arXiv preprint
- **Key Idea:** VAE with masked reconstruction and spatiotemporal trajectory modeling.
- **Relevance:** Related VAE architecture innovation.

---

## Topic B: VAE Semantic Drift / Latent Space Semantic Alignment

### TOP PRIORITY

#### 1. SFD (CVPR 2026) — already covered in Topic A

#### 2. DA-VAE — Detail-Aligned VAE ⭐⭐⭐
- **Venue:** CVPR 2026
- **Key Idea:** Latent compression via detail alignment. Training-time VAE improvement.
- **Relevance:** Addresses reconstruction quality vs semantic fidelity in VAE.

#### 3. VAE-SRA / SRA2 — VAE Self-Representation Alignment ⭐⭐⭐
- **arXiv:** 2601.17830 (January 2026)
- **Key Idea:** VAE self-representation alignment for latent space structure.
- **Relevance:** Related to VAE latent space semantic structure.

#### 4. PS-VAE — Pixel-Semantic VAE ⭐⭐
- **Date:** December 2025
- **Key Idea:** Addresses off-manifold diffusion artifacts via pixel-semantic decomposition.
- **Relevance:** Semantic-perceptual disentanglement in VAE.

#### 5. Semantic-VAE ⭐⭐
- **arXiv:** 2509.22167 (September 2025)
- **Key Idea:** Semantic alignment for speech synthesis.
- **Code:** Available
- **Relevance:** Semantic alignment method applicable to VAE.

---

## Key Strategic Insights

### TrACE-Video Position:
- LIPAR validates latent space has exploitable structure (supports VAE latent consistency premise)
- Pathwise TTC provides a CORRECTION mechanism (TrACE-Video detects → TTC corrects)
- SFD provides complementary GENERATION approach (semantic-first)
- TrACE-Video's unique contribution: unsupervised measurement methodology without training

### CNLSA Position:
- SFD confirms semantic drift is a recognized problem (CVPR 2026 paper)
- Multiple VAE improvement approaches (DA-VAE, VAE-SRA, PS-VAE) confirm this is an active research area
- CNLSA's cross-encoder validation methodology is novel (no other paper does cross-encoder confound identification)

### Gap Identified:
No paper combines: (1) unsupervised latent consistency metric + (2) cross-encoder validation + (3) integration with test-time correction (TTC). This is TrACE-Video's niche.

---

## Updated Paper Priority for Next Steps

1. **LIPAR** (2603.05811) — Read code, understand latent patch redundancy detection method
2. **Pathwise TTC** (2602.05871) — Read code, understand test-time correction mechanism
3. **SFD** (2512.04926) — Read, understand async semantic denoising
4. **Long-Horizon SVG** (2604.10103) — Scan for methodology
5. **DA-VAE** — Check if code available
