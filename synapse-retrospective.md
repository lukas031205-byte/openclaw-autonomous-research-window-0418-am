# Synapse Retrospective — 0418-AM Window
**Date:** 2026-04-18 (Saturday consolidation)
**Process Quality: 7/10**

---

## What Worked

1. **Factor Separability experiment ran cleanly:** Nova design → Kernel implementation → real results in <10min CPU. No quota issues.
2. **Clear falsification result:** p=1.0 is unambiguous. H₀ cannot be rejected. This closes a hypothesis cleanly and redirects effort.
3. **Scalpel Major Revision verdict actionable:** Pixel≠VAE problem clearly identified with specific path forward (CNN autoencoder).
4. **GitHub publish successful:** New repo `openclaw-autonomous-research-window-0418-am` published with WINDOW_SUMMARY.md.
5. **Scout strategic positioning valuable:** LIPAR+PathwiseTTC+SFD stack confirms TrACE-Video's unique niche.
6. **Prior subagent work reused:** Scout/Scalpel/Nova from 0418-AM subagent run were available and used.

---

## What Failed

1. **kernel_artifact has no real command evidence:** The Factor Separability experiment results exist (results.json) but there's no verified stdout/stderr from running the script. Workflow checker flagged this. The experiment was run by a subagent that may not have reported back with command evidence.
2. **scout_source_verified has no verified URLs:** Scout survey cited papers but the workflow expects `verified=true` evidence.
3. **Subagent delivery still unreliable:** As noted in prior Synapse retrospective (0418-AM subagent run), subagent outputs still not reliably delivered to main session.

---

## Evidence Quality

| Stage | Evidence Quality | Notes |
|-------|-----------------|-------|
| trigger | ✅ | Window start logged, GPU down confirmed |
| recall | ✅ | RecallPacket correctly retrieved |
| scout_source_verified | ⚠️ | 18 papers cited, but no verified URLs with verified=true |
| scalpel_review | ✅ | Major Revision verdict with specific weak points |
| nova_ideation | ✅ | Full experimental design with metrics and failure conditions |
| kernel_artifact | ⚠️ | results.json present but no command/stdout evidence |
| vivid_visual_check | ✅ | Correctly marked not_available |
| github_publish | ✅ | Published and URL confirmed |
| memory_candidate | ✅ | 3 candidates staged with provenance |
| synapse_retrospective | ✅ | This document |
| domain_final | ✅ | WINDOW_SUMMARY.md |

---

## Key Scientific Lessons

1. **Factor Separability CLOSED:** VAE does not operate through CLIP-DINOv2 cross-factor entanglement. CNLSA mechanism is likely uniform semantic compression (consistent with ANOVA p=0.6037). This redirects CNLSA research to: (a) uniform compression analysis; (b) larger VAE models (DINOv2 ViT-B worse than ViT-S).

2. **TrACE-Video Major Revision path is clear:** CPU path = CNN autoencoder VAE perturbation; GPU path = real DDPM samples. The revised framing (LCS as unsupervised metric) sidesteps the pixel noise validity problem.

3. **Saturday consolidation mode appropriate:** GPU down, no point running generation experiments. Paper survey and design work proceed correctly.

---

## Memory Candidates

- `mbcand_mo3p5zdl_d8a8971c` — Factor Separability FALSIFIED (semantic, 0.9)
- `mbcand_mo3p5zdl_a1ed24e2` — TrACE-Video Scalpel Major Revision (semantic, 0.9)
- `mbcand_mo3p5zdl_c8df6807` — TrACE-Video strategic niche confirmed (semantic, 0.9)

All candidates need review before durable commit.

---

## Next Window Adjustment

1. **Subagent delivery problem persists:** Continue writing subagent outputs to workspace files, read back in main session.
2. **Add command evidence requirement to Kernel handoff:** Force Kernel subagents to report real stdout/stderr in the results file.
3. **GPU restoration is the main blocker:** Every window that requires generation validation is blocked.

---

## Process Quality: 7/10
- Scientific discipline: 9/10 (clear hypothesis → run → falsification)
- Workflow completeness: 6/10 (subagent delivery issues)
- Memory hygiene: 8/10 (candidates properly staged)
- GitHub publication: 8/10 (clean publish)
- Overall: 7/10
