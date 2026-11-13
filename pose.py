"""Pose-control attack: Induced Misdirection (IM).

The target pose x_p handed to the model is replaced by a pose S_r drawn at
random from a pool of poses, and a perturbation is added:

    S_g = S_r + eps                                                 

The result is clamped to the valid range of the pose maps so it can still be
visualised. The source image is left untouched.

In the original experiments the pool was the folder of .npy pose files
(PIDM/data/deepfashion_256x256/target_pose) and the pose was picked with
``random.choice``; ``random_poses_from_pool`` does the same. When no folder is
given, the pool is the set of target poses of the other evaluated pairs
(``induced_misdirection``).
"""
from __future__ import annotations

import numpy as np
import torch


def misdirection_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """For each of n samples, the index of a different, randomly chosen sample."""
    if n < 2:
        raise ValueError("Induced Misdirection needs at least two test pairs")
    idx = rng.integers(0, n - 1, size=n)
    idx[idx >= np.arange(n)] += 1  # skip the sample's own pose
    return idx


def induced_misdirection(
    pose_pool: torch.Tensor,
    rng: np.random.Generator,
    epsilon: float = 0.0,
) -> tuple[torch.Tensor, np.ndarray]:
    """Return the misdirected pose for every sample in ``pose_pool``.

    pose_pool  (N, 20, H, W) target-pose maps of the evaluated test pairs
               (3 skeleton channels + 17 limb distance maps, as in PIDM).
    epsilon    std of the Gaussian noise added to the selected pose maps.
    Returns (poses, idx) where poses[i] = clamp(pose_pool[idx[i]] + eps).
    """
    idx = misdirection_indices(pose_pool.shape[0], rng)
    poses = pose_pool[torch.as_tensor(idx)].clone()
    if epsilon > 0:
        noise = rng.normal(0.0, epsilon, size=tuple(poses.shape)).astype(np.float32)
        lo, hi = float(pose_pool.min()), float(pose_pool.max())
        poses = (poses + torch.from_numpy(noise)).clamp(lo, hi)
    return poses, idx


def random_poses_from_pool(
    pool: torch.Tensor,
    n: int,
    rng: np.random.Generator,
    epsilon: float = 0.0,
) -> tuple[torch.Tensor, np.ndarray]:
    """Draw n poses uniformly at random from ``pool`` (K, 20, H, W)."""
    idx = rng.integers(0, pool.shape[0], size=n)
    poses = pool[torch.as_tensor(idx)].clone()
    if epsilon > 0:
        noise = rng.normal(0.0, epsilon, size=tuple(poses.shape)).astype(np.float32)
        poses = (poses + torch.from_numpy(noise)).clamp(float(pool.min()), float(pool.max()))
    return poses, idx
