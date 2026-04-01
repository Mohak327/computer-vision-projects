# PhysNeRF-BS v1 Project Plan

## Summary
- Build a Colab-friendly, package-and-scripts research prototype of PhysNeRF-BS around the standard `dm_control` humanoid, using scripted motion primitives as the canonical data source.
- Deliver the full v1 stack: MuJoCo data generation, pose-conditioned NeRF, hybrid Hamiltonian regularization, a separate motor-gated neural field world model, simulator-native evaluation, and a thin future CLARISNet adapter.
- Treat the current notebooks as reference implementations only. The 2D neural-field ideas from `main_part1.ipynb` and NeRF utilities from `main_part2.ipynb` should be extracted into reusable modules rather than extended in-place.

## Module Plan
1. `config`
- Use YAML configs plus lightweight Python dataclasses for all experiments so runs work both from CLI and Colab without Hydra-heavy indirection.
- Define config groups for simulation, dataset, NeRF training, world-model training, physics loss weights, and evaluation.

2. `sim`
- Wrap `dm_control` humanoid loading, camera rig creation, headless rendering, and deterministic seeding behind a simulator service.
- Implement scripted motion generators for `walk`, `arm_raise`, and `crouch`, plus pose interpolation and mild noise for coverage.
- Log `qpos`, `qvel`, control/torque, timestamps, camera extrinsics/intrinsics, and MuJoCo-derived energy terms per frame.

3. `data`
- Build one unified dataset writer that emits both multiview reconstruction data and temporal sequence data from the same simulation runs.
- Store RGB, depth, segmentation, and metadata manifests so NeRF, physics losses, and the world model consume the same source of truth.
- Add dataset readers for ray batches, pose-conditioned frame batches, and sequential visuomotor clips.

4. `models.nerf`
- Start from the existing NeRF notebook implementation and convert it into a pose-conditioned dynamic NeRF with a joint-state encoder.
- Keep hierarchical sampling, chunked rendering, and coarse/fine structure, but make pose conditioning explicit at every queried sample.
- Support training modes for baseline static NeRF, pose-conditioned NeRF, and pose-conditioned NeRF plus physics loss.

5. `physics`
- Implement a hybrid Hamiltonian module where MuJoCo teacher energy is the primary target and geometry-derived proxies from predicted depth are auxiliary regularizers.
- Compute kinetic and potential consistency from simulator state, with optional depth-based center-of-mass and silhouette plausibility penalties.
- Expose losses cleanly so they can be toggled for ablations without changing trainer code.

6. `models.world_model`
- Rebuild the motor-gated neural field as a separate temporal model using Amari-style recurrent field dynamics, motor gating, and a visual decoder.
- Train it on visuomotor prediction over clips rather than isolated frames.
- Keep this loosely coupled to NeRF in v1: same dataset, separate trainer, shared evaluation comparisons, no joint optimization.

7. `training`
- Provide one trainer for NeRF experiments and one for the world model, each with checkpointing, validation hooks, and metric logging.
- Add a thin experiment orchestrator that runs the recommended order: dataset generation, static NeRF baseline, pose-NeRF, pose-NeRF plus physics, then world-model training.

8. `evaluation`
- Implement visual metrics: PSNR, SSIM, LPIPS, and explicit novel-view splits.
- Implement simulator-native physics metrics: energy consistency error, temporal smoothness, and occluded-limb reconstruction sensitivity.
- Implement body-selectivity analysis for the motor-gated channels using masked body/background regions from segmentation renders.
- Keep Isaac-Gym-style stability evaluation out of the mandatory path; leave it as an optional extension module.

9. `adapters`
- Add a thin CLARISNet-facing contract that can later consume external joint-state streams instead of MuJoCo logs.
- Keep this adapter small: input normalization, schema validation, and conversion into the shared dataset/training batch types.

10. `scripts`
- Provide script entrypoints for `generate_dataset`, `train_nerf`, `train_world_model`, `evaluate_nerf`, `evaluate_world_model`, and `run_ablation_suite`.
- Make every script callable from both terminal and Colab with config path overrides and output directory selection.

## Public Interfaces and Types
- Define a `FrameRecord` schema with image paths, depth path, segmentation path, `qpos`, `qvel`, torque/control, timestamp, camera id, camera matrices, motion id, and split.
- Define a `SequenceRecord` schema as an ordered list of frame ids plus the aligned motor stream for world-model training.
- Define a `PhysicsTargets` type that carries simulator energy terms and any enabled depth-derived proxy targets.
- Define a `ClarisNetAdapterIO` contract that accepts external joint-state sequences and emits the same normalized records used by the MuJoCo pipeline.
- Standardize model outputs so NeRF returns RGB, depth, density statistics, and loss-ready intermediates, while the world model returns predicted frames, latent field state, and gating diagnostics.

## Test Plan
- Dataset smoke test: generate a tiny run with one motion, two cameras, and confirm manifests, image files, and tensor shapes are valid.
- Determinism test: same seed should reproduce motion trajectories, split assignment, and camera setup.
- NeRF unit tests: ray sampling, positional encoding dimensions, hierarchical rendering shapes, and pose-conditioning plumbing.
- Physics tests: teacher-energy calculations should match simulator values on logged states, and proxy losses should change when depth or pose is perturbed.
- World-model tests: motor gating should accept aligned torque streams and produce stable recurrent rollout shapes over short clips.
- End-to-end smoke run: one mini config should execute on Colab and produce checkpoints plus evaluation JSON for each stage.

## Assumptions and Defaults
- Use the standard `dm_control` humanoid as the surrogate body for v1 rather than building a custom 17-DOF asset first.
- Use scripted motions as the primary dataset source; no learned controller is required in the initial build.
- Use loose coupling between NeRF and the neural field in v1; comparison and shared data are required, joint training is not.
- Use hybrid teacher physics: simulator-derived energy and joint dynamics are primary, geometry-derived terms are auxiliary, not the only supervision.
- Use simulator-native physics evaluation in the required path; external stability benchmarking is optional.
- Target a clean Python package with scripts as the source of truth, while preserving notebooks only as references or demos.
