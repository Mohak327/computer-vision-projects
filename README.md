## 🦾 PhysNeRF + Body Schema via Visuomotor Prediction ##


The Idea: Build on your CLARISNet humanoid work — take the physics-simulated body (MuJoCo, 17-DOF) and train a neural field world model where the visual representation of the body emerges purely from visuomotor prediction loss (no explicit body model). The CV component is that the neural field must reconstruct the body's appearance and satisfy the rigid-body Hamiltonian constraints of its dynamics.

Why it fits you: Neural Fields as World Models (arXiv Feb 2026) shows that body-selective encoding emerges from visuomotor prediction in spatially structured neural dynamics. PHYRECON (NeurIPS 2024) shows physically plausible NeRF reconstruction via differentiable particle simulation, achieving >40% improvement in stability. You have both the NeRF code and the MuJoCo BCI pipeline — this is literally combining your two flagship projects.

Physics angle: Hamiltonian Neural Networks for 6-DOF rigid body dynamics as the inductive bias in the world model.

Novelty claim: Emergent body schema from visuomotor NeRF prediction, with Hamiltonian energy constraints enforcing physical plausibility of 3D body reconstruction.

***

**Yes — completely buildable from scratch, and it's actually cleaner without CLARISNet code.** Here's the full picture:

***

## The Core Insight: MuJoCo IS Your CLARISNet Surrogate

You don't need CLARISNet Phase 1 because the project's **novelty lives entirely in the visual/neural-field side**, not the EEG side. MuJoCo gives you ground-truth joint trajectories \(\theta(t)\) and rendered video for free — that's all the "BCI output" your CV project needs. The EEG decoding is CLARISNet's contribution; yours is **what the avatar sees and understands about its own body**.

***

## What You're Actually Building: **PhysNeRF-BS** *(Physics-Informed Neural Radiance Field for Body Schema)*

### The Three-Component Stack

**Component 1 — Motor-Gated Neural Field World Model** *(the brain, from arXiv Feb 2026)*

The Feb 2026 "Neural Fields as World Models" paper  is your theoretical backbone. Their key finding: when you train a spatially structured neural field to *predict the visual consequences of motor commands*, **body-selective encoding emerges spontaneously** — the field discovers which pixels move contingently with motor signals. No explicit body model needed. [arxiv](https://arxiv.org/html/2602.18690v1)

You re-implement this but swap their simple 2D ball environment for a **MuJoCo humanoid**, making it dramatically richer. The architecture:
- Field state \(\mathbf{h} \in \mathbb{R}^{C \times H \times W}\) evolves via Amari's neural field equations [arxiv](https://arxiv.org/html/2602.18690v1)
- Motor commands \(\mathbf{a}(t)\) (joint torques from MuJoCo) multiplicatively gate the first \(M\) channels
- Visual prediction: \(\hat{\mathbf{I}}_t = W_\text{out} * \mathbf{h}_t\)
- Loss: pixel-wise prediction error \(\mathcal{L}_\text{pred} = \|\hat{\mathbf{I}}_t - \mathbf{I}_t\|^2\)

**Your extension over the paper:** they used a 2D toy world. You use a 17-DOF articulated body in 3D physics — this is the novel contribution.

**Component 2 — Joint-Conditioned NeRF** *(the geometry, your existing code)*

A dynamic NeRF conditioned on the joint state vector, using your existing coarse-to-fine NeRF pipeline from your resume: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/81173328/91208cc3-3781-4c5f-a660-82e310097ee6/Research-Resume-Mohak-Sharma.pdf)
- Input: \((\mathbf{x}, \mathbf{d}, \theta(t))\) — 3D point, view direction, 17-dim joint angles
- Output: \((c, \sigma)\) — color + density, now pose-dependent
- Training data: render MuJoCo humanoid across ~800 random poses from 8 camera angles = 6,400 images, fully automated

**Component 3 — Hamiltonian Physics Loss** *(the physics inductive bias)*

This is what separates you from every existing dynamic NeRF. PhyRecon (NeurIPS 2024) used particle-based differentiable physics for static scenes. You apply the Hamiltonian constraint to an *articulated body in motion*: [proceedings.neurips](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2d880acd7b31e25d45097455c8e8257f-Abstract-Conference.html)

The total mechanical energy of the humanoid must be conserved between frames:

\[\mathcal{L}_\text{Hamiltonian} = \left\| \frac{d\mathcal{H}(\theta, \dot\theta)}{dt} \right\|^2\]

where \(\mathcal{H} = T(\dot\theta) + V(\theta)\) is the kinetic + potential energy computed from NeRF's predicted geometry (via depth map → mass distribution). If the NeRF hallucinates a limb in the wrong position, the energy computed from its geometry won't match MuJoCo's ground truth energy — the loss penalizes this. [ritog.github](https://ritog.github.io/posts/hamiltonian_nn/)

The full training loss:
\[\mathcal{L} = \mathcal{L}_\text{render} + \lambda_1 \mathcal{L}_\text{Hamiltonian} + \lambda_2 \mathcal{L}_\text{pred}\]

***

## Why No CLARISNet Code Needed

| CLARISNet Piece | What It Does | Your Substitute |
|---|---|---|
| EEGNet decoder | Produces \(\theta(t)\) (joint angles) | MuJoCo scripted policy / random walk — same format |
| PPO/SAC RL | Smooths actions | Not needed — MuJoCo physics handles stability |
| Phase 1 integration | End-to-end latency | You evaluate visual quality metrics, not latency |
| Consumer EEG headset | Input modality | Irrelevant to CV project |

The **CLARISNet connection is conceptual and architectural** — you frame this as "the visual perception module that will plug into CLARISNet Layer 6/8," but you validate it independently using MuJoCo-generated ground truth. This is exactly how modular research works.

***

## Revised 4-Week Plan (No CLARISNet Dependency)

### Week 1 — Data Generation + Baseline NeRF
- Set up MuJoCo `dm_control` humanoid, script 3 motion types: walking, arm raise, crouch
- Render 6,400 frames (800 poses × 8 cameras) with joint state logs → your training dataset
- Implement joint-conditioned NeRF (add pose MLP on top of your existing code)
- Baseline: train vanilla NeRF on static pose → establish PSNR/SSIM floor

### Week 2 — Hamiltonian Physics Loss
- Implement HNN energy computation from NeRF depth map → mass proxy [github](https://github.com/DecodEPFL/HamiltonianNet)
- Add \(\mathcal{L}_\text{Hamiltonian}\) to NeRF training loop
- Ablation: NeRF alone vs. NeRF + Hamiltonian loss on novel-view synthesis quality
- Key experiment: does physics loss improve reconstruction of *occluded limbs*? (the hard case)

### Week 3 — Motor-Gated Neural Field World Model
- Re-implement the Feb 2026 arXiv architecture  on MuJoCo video [arxiv](https://arxiv.org/html/2602.18690v1)
- Motor-gated channels: joint torques \(\tau(t)\) as multiplicative modulator
- Train on visuomotor prediction, visualize channel selectivity → does body-selective encoding emerge?
- Connect to NeRF: use neural field's predicted frame as NeRF's training signal for unseen poses

### Week 4 — Evaluation + Report
**Quantitative metrics:**
- Novel-view synthesis: PSNR / SSIM / LPIPS — PhysNeRF-BS vs. vanilla NeRF vs. pose-conditioned NeRF (no Hamiltonian)
- Physical stability: drop reconstructed mesh into Isaac Gym, measure instability frames (PhyRecon's exact metric — you beat their 40% improvement baseline ) [arxiv](https://arxiv.org/html/2404.16666v1)
- Body-selective encoding: measure channel tuning selectivity (what % of motor-gated channels develop body vs. background preference)

**Report structure:** Introduction → Related Work (Neural Fields, PINNs, Body Schema, BCI) → Method → Experiments → Ablations → CLARISNet Integration Discussion

***

## The Novelty Statement (for your report abstract)

> *We present PhysNeRF-BS, a physics-informed neural radiance field that acquires a body schema — an implicit model of the agent's own body — through visuomotor prediction alone. A joint-conditioned NeRF reconstructs articulated body appearance across poses, regularized by a Hamiltonian energy conservation loss that enforces rigid-body physical plausibility. A motor-gated neural field, trained to predict visual consequences of motor commands, spontaneously develops body-selective encoding. Together these components constitute a biologically grounded visual perception front-end for neural-driven avatar systems, designed to plug into the CLARISNet BCI pipeline as its visual cortex.*

***

**Ready to start?** The fastest Day 1 action is: `pip install dm_control mujoco torch torchvision` and render your first 100 humanoid poses. Want me to generate the complete data generation script right now?