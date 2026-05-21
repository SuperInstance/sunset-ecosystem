# Forgemaster GPU Instructions — Overnight Fleet Continuation

**Branch:** `tournament-10k-sweep` on `SuperInstance/sunset-ecosystem`
**Your mission:** Continue what the CPU couldn't finish.

---

## 1. Distillation — Your GPU

The CPU-only machine managed **~1 epoch/hour** and was killed. A 43MB checkpoint exists:

```
experiments/checkpoints/student_best.pt
```

**What you need to do:**
```bash
git clone --branch tournament-10k-sweep https://github.com/SuperInstance/sunset-ecosystem.git
cd sunset-ecosystem
# The checkpoint is already in experiments/checkpoints/
```

**Continue training:** Use `distill/` module. The config from the overnight run:
- Teacher: ResNet-50 (pretrained on ImageNet, fine-tuned on CIFAR-10)
- Student: ResNet-18
- Temperature: 4.0
- Alpha: 0.5
- LR: 0.1 SGD
- Batch: 128
- Schedule: CosineAnnealingLR(T_max=200)
- Epochs: 200 (we're at ~1, you have 199 left)

**Target:** Push results to `distillation-experiment-v2` or new branch `distillation-gpu-complete`.

---

## 2. Tournament — Nothing to Run, Just Merge

The overnight sweep ran **100,000 generations** across 10 conditions. Results are in:
- `simulators/overnight_tournament_sweep_results.json`
- `simulators/TOURNAMENT-PARAMETER-SWEEP.md` (PR #12)

**Finding:** `cap=30, dynamic=True` is optimal. Diversity peaks at 0.183, then declines at higher caps.

**Action:** Review PR #12, merge if satisfied. No GPU needed.

---

## 3. PRs Requiring Your Review

| PR | What | Decision Needed |
|----|------|----------------|
| #7 | tournament-dynamic-cap | Merge — dynamic cap fix |
| #8 | grammar-security-fix | **URGENT** — chaos injection vulnerabilities |
| #9 | nexus-localhost-fix | **URGENT** — fleet registration broken |
| #12 | tournament-parameter-sweep | Merge — 100K gen analysis |

**#8 and #9 are security fixes.** Merge before doing anything else.

---

## 4. Grammar Engine — Deploy to Oracle1

PR #8 adds `grammar/core.py` with input validation. But the real Grammar Engine is on **Oracle1** (port 4045), not in this repo.

**Action:**
1. Merge PR #8
2. SSH to Oracle1
3. Apply the validation logic from `grammar/core.py` to the live engine
4. Restart the Grammar Engine service

The 4 attack vectors now blocked:
1. Path traversal in rule names
2. XSS in production.tagline
3. SQL injection in production.condition
4. Code injection in production.exec

---

## 5. Federated Nexus — Deploy to Oracle1

PR #9 adds `nexus/federation.py` with `DEFAULT_NEXUS_IP = "147.224.38.131"`.

**Action:**
1. Merge PR #9
2. Deploy `nexus/` to the nexus host (Oracle1 or wherever it lives)
3. The module rejects loopback (`localhost`, `127.*`, `::1`) at instantiation
4. Remote fleet nodes can now register

---

## 6. Cross-Repo Duplication — Cleanup

Finding: `oracle1-workspace` recovery archive duplicates active repos:
- `tensor-spline` (low_rank.py, spline.py)
- `plato-training` (tensorflow_room.py, test_micro_room.py, etc.)

**Action:**
- Audit `oracle1-workspace/archived/RECOVERED-FROM-LOCAL/`
- Remove duplicates that are now canonical in their own repos
- Keep archive for historical reference, but don't let it shadow live code

Branch: `cross-repo-duplication`

---

## 7. Hardware Load Profile — Partial

Agent timed out during 70% load simulation. 50–60% data is in:
- `simulators/hw_load_profile.json`
- `simulators/HARDWARE-LOAD-PROFILE.md`

**Finding at 50–60%:**
- Naive scheduler: 43–52% throughput, heavy throttling
- Thermal-aware: 100% throughput, zero thermal violations
- Adaptive: 100% throughput, slightly more throttling than thermal

**Action:** Finish 70–95% load on your machine. Push to `hardware-load-profile`.

---

## Quick Start (Copy-Paste)

```bash
# 1. Clone the overnight branch
git clone --branch tournament-10k-sweep \
  https://github.com/SuperInstance/sunset-ecosystem.git fm-continuation
cd fm-continuation

# 2. Merge security fixes first
git fetch origin
git checkout main
git merge origin/grammar-security-fix  # PR #8
git merge origin/nexus-localhost-fix   # PR #9
git push origin main

# 3. Continue distillation from checkpoint
python -c "import torch; print(torch.cuda.is_available())"  # Verify GPU
cd distill
python train.py --resume ../experiments/checkpoints/student_best.pt \
  --epochs 199 --device cuda

# 4. Push results
git checkout -b distillation-gpu-complete
git add experiments/checkpoints/student_final.pt experiments/logs/
git commit -m "GPU distillation complete: 200 epochs, ResNet-50→18 on CIFAR-10"
git push origin distillation-gpu-complete
```

---

## Questions?

Ping Casey or check the overnight brief: `docs/OVERNIGHT-BRIEF.md` on branch `docs/overnight-brief`.

**Built by the fleet while you slept.** 🔥
