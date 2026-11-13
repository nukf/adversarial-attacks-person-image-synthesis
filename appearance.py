r"""Robustness of PIDM to input perturbations.

Runs attacks against a pretrained PIDM model
(https://github.com/ankanbhunia/PIDM, "Person Image Synthesis via Denoising
Diffusion Model").

For every source image ``input{i}.jpg`` and target pose ``pose{i}.npy`` the
script generates a clean baseline (``B1.jpg``) and one output per selected
attack (``B1_<name>.jpg``), saved to ``<output-dir>/<i>/``.

Appearance-control attacks (the source image is perturbed):
    F1-F11   Fourier-domain salt & pepper / Gaussian / uniform noise
    I1-I6    contrast and brightness changes
    GA1-GA3  Gaussian noise
    GH1-GH5  ghosting (shifted / rotated / scaled copies)
    C1-C2    chromatic aberration

Pose-control attack (the source image is left untouched):
    IM       Induced Misdirection. The target pose x_p is replaced by a
             randomly drawn pose S_r plus Gaussian noise eps, S_g = S_r + eps,
             clamped to the valid range of the pose maps. S_r comes from the
             other evaluated pairs, or from the folder given by
             --pose-pool-dir. Set the noise std with --epsilon.

Examples:
    python attack_on_pidm.py --pidm-root /path/to/PIDM \
        --input-dir /path/to/target_edits --pose-dir /path/to/target_pose \
        --output-dir /path/to/results --start 1 --end 40

"""

from __future__ import annotations

import argparse
import csv
import sys
import zlib
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


# ---------------------------------------------------------------------------
# Perturbations
# Each takes a uint8 BGR image (OpenCV convention) and returns a new uint8 BGR
# image of the same shape. Randomness comes from an explicit numpy Generator.
# ---------------------------------------------------------------------------

def freq_perturb(image, strength, mode, rng=None):
    """Perturb each colour channel in the Fourier domain.

    Args:
        image: uint8 BGR image.
        strength: noise scale. For ``salt_pepper`` it sets the fraction of
            frequency bins overwritten (``0.1 * strength / 10`` each for salt
            and pepper); for ``gaussian`` it is the standard deviation; for
            ``random`` it is the half-width of the uniform distribution.
        mode: ``"salt_pepper"``, ``"gaussian"`` or ``"random"``.
        rng: optional ``numpy.random.Generator``.
    """
    if mode not in ("salt_pepper", "gaussian", "random"):
        raise ValueError(f"Unknown frequency perturbation mode: {mode!r}")
    rng = rng or np.random.default_rng()

    channels = []
    for channel in cv2.split(image):
        spectrum = np.fft.fft2(channel)

        if mode == "salt_pepper":
            prob = 0.1 * (strength / 10)
            spectrum[rng.random(spectrum.shape) < prob] = 255  # salt
            spectrum[rng.random(spectrum.shape) < prob] = 0  # pepper
        elif mode == "gaussian":
            spectrum += rng.normal(0, strength, spectrum.shape)
        else:  # "random"
            spectrum += rng.uniform(-strength, strength, spectrum.shape)

        restored = np.fft.ifft2(spectrum).real
        channels.append(np.clip(restored, 0, 255).astype(np.uint8))

    return cv2.merge(channels)


def intensity_transform(image, contrast, brightness, rng=None):
    """Linear intensity change: ``|contrast * pixel + brightness|``, saturated."""
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)


def gaussian_perturb(image, sigma, rng=None):
    """Add zero-mean pixel-space Gaussian noise with standard deviation ``sigma``."""
    rng = rng or np.random.default_rng()
    noisy = image.astype(np.float32) + rng.normal(0, sigma, image.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def ghosting(image, num_ghosts, opacity, shift_range, rotation_range, scale_range, rng=None):
    """Blend randomly shifted / rotated / scaled copies of the image onto itself.

    Args:
        num_ghosts: number of ghost copies.
        opacity: blend weight of each ghost (0-1).
        shift_range: max translation in pixels, per axis.
        rotation_range: max rotation in degrees (0 disables).
        scale_range: max relative scale change (0 disables).
    """
    rng = rng or np.random.default_rng()
    height, width = image.shape[:2]
    ghosted = image.copy()

    for _ in range(num_ghosts):
        shift = rng.integers(-shift_range, shift_range, endpoint=True, size=2)
        angle = rng.uniform(-rotation_range, rotation_range) if rotation_range > 0 else 0.0
        scale = 1 + rng.uniform(-scale_range, scale_range) if scale_range > 0 else 1.0

        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
        matrix[:, 2] += shift
        ghost = cv2.warpAffine(image, matrix, (width, height))
        cv2.addWeighted(ghost, opacity, ghosted, 1 - opacity, 0, ghosted)

    return ghosted


def chroma(image, shift_r, shift_g, shift_b, rng=None):
    """Chromatic aberration: roll each colour channel vertically by its own offset."""
    b, g, r = cv2.split(image)
    return cv2.merge(
        [
            np.roll(b, shift_b, axis=0),
            np.roll(g, shift_g, axis=0),
            np.roll(r, shift_r, axis=0),
        ]
    )


# Experiment registry: name -> f(image, rng=rng) -> perturbed image
ATTACKS = {
    # Frequency-domain salt & pepper
    "F1": partial(freq_perturb, strength=5, mode="salt_pepper"),
    "F2": partial(freq_perturb, strength=10, mode="salt_pepper"),
    "F3": partial(freq_perturb, strength=20, mode="salt_pepper"),
    # Frequency-domain Gaussian
    "F4": partial(freq_perturb, strength=5, mode="gaussian"),
    "F5": partial(freq_perturb, strength=10, mode="gaussian"),
    "F6": partial(freq_perturb, strength=15, mode="gaussian"),
    "F7": partial(freq_perturb, strength=30, mode="gaussian"),
    # Frequency-domain uniform
    "F8": partial(freq_perturb, strength=5, mode="random"),
    "F9": partial(freq_perturb, strength=10, mode="random"),
    "F10": partial(freq_perturb, strength=20, mode="random"),
    "F11": partial(freq_perturb, strength=50, mode="random"),
    # Intensity (contrast, brightness)
    "I1": partial(intensity_transform, contrast=0.5, brightness=0),
    "I2": partial(intensity_transform, contrast=3.0, brightness=0),
    "I3": partial(intensity_transform, contrast=1.0, brightness=-30),
    "I4": partial(intensity_transform, contrast=1.0, brightness=50),
    "I5": partial(intensity_transform, contrast=2.0, brightness=50),
    "I6": partial(intensity_transform, contrast=1.5, brightness=30),
    # Pixel-space Gaussian noise
    "GA1": partial(gaussian_perturb, sigma=5),
    "GA2": partial(gaussian_perturb, sigma=10),
    "GA3": partial(gaussian_perturb, sigma=20),
    # Ghosting
    "GH1": partial(ghosting, num_ghosts=3, opacity=0.5, shift_range=10, rotation_range=0, scale_range=0),
    "GH2": partial(ghosting, num_ghosts=10, opacity=0.7, shift_range=10, rotation_range=0, scale_range=0),
    "GH3": partial(ghosting, num_ghosts=50, opacity=0.5, shift_range=10, rotation_range=360, scale_range=0.2),
    "GH4": partial(ghosting, num_ghosts=20, opacity=0.4, shift_range=15, rotation_range=180, scale_range=0.1),
    "GH5": partial(ghosting, num_ghosts=100, opacity=0.6, shift_range=20, rotation_range=360, scale_range=0.3),
    # Chromatic aberration
    "C1": partial(chroma, shift_r=5, shift_g=10, shift_b=15),
    "C2": partial(chroma, shift_r=15, shift_g=20, shift_b=25),
}


# ---------------------------------------------------------------------------
# Pose-control attack: Induced Misdirection (IM)
#
# The target pose x_p handed to the model is replaced by a pose S_r drawn at
# random from a pool of poses, and a perturbation is added:
#
#     S_g = S_r + eps
#
# The result is clamped to the valid range of the pose maps so it can still be
# visualised. Poses are numpy arrays of shape (20, H, W): 3 skeleton channels
# + 17 limb distance maps, as in PIDM. They are loaded one at a time, so the
# pool can be the whole pose folder without holding it in memory.
# ---------------------------------------------------------------------------

def load_pose(path: Path) -> np.ndarray:
    """Load a pose ``.npy`` file as a (C, H, W) array, as PIDM's pipeline does."""
    return T.ToTensor()(np.load(path)).numpy()


def misdirection_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """For each of n samples, the index of a different, randomly chosen sample."""
    if n < 2:
        raise ValueError("Induced Misdirection needs at least two test pairs")
    idx = rng.integers(0, n - 1, size=n)
    idx[idx >= np.arange(n)] += 1  # skip the sample's own pose
    return idx


def pose_value_range(paths: list[Path]) -> tuple[float, float]:
    """Smallest and largest value over all pose maps in ``paths``."""
    lo, hi = np.inf, -np.inf
    for path in paths:
        pose = load_pose(path)
        lo, hi = min(lo, float(pose.min())), max(hi, float(pose.max()))
    return lo, hi


def choose_misdirection_poses(
    eval_pose_paths: list[Path],
    pool_paths: list[Path] | None,
    rng: np.random.Generator,
) -> list[Path]:
    """Pick the pose S_r handed to the model for every evaluated sample.

    Without ``pool_paths`` the pool is the target poses of the other evaluated
    pairs (each sample gets a pose that is not its own). With ``pool_paths``
    poses are drawn uniformly at random from that pool, as ``random.choice``
    did in the original experiments, so a sample may occasionally get its own pose.
    """
    if pool_paths is None:
        idx = misdirection_indices(len(eval_pose_paths), rng)
        return [eval_pose_paths[j] for j in idx]
    idx = rng.integers(0, len(pool_paths), size=len(eval_pose_paths))
    return [pool_paths[j] for j in idx]


def add_pose_noise(
    pose: np.ndarray,
    epsilon: float,
    bounds: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """S_g = clamp(S_r + eps), eps ~ N(0, epsilon^2), clamped to ``bounds``."""
    if epsilon <= 0:
        return pose.copy()
    noise = rng.normal(0.0, epsilon, size=pose.shape).astype(np.float32)
    return np.clip(pose + noise, *bounds)


# ---------------------------------------------------------------------------
# Model and experiment
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pidm-root", type=Path, default=Path("."), help="Path to the PIDM repository.")
    p.add_argument("--config", type=Path, help="Diffusion config (default: <pidm-root>/config/diffusion.conf).")
    p.add_argument("--checkpoint", type=Path, help="Model checkpoint (default: <pidm-root>/checkpoints/last.pt).")
    p.add_argument("--input-dir", type=Path, required=True, help="Folder with input{i}.jpg source images.")
    p.add_argument("--pose-dir", type=Path, required=True, help="Folder with pose{i}.npy target poses.")
    p.add_argument("--output-dir", type=Path, required=True, help="Where results are written (one sub-folder per sample).")
    p.add_argument("--start", type=int, default=1, help="First sample index (inclusive).")
    p.add_argument("--end", type=int, default=40, help="Last sample index (inclusive).")
    p.add_argument(
        "--attacks",
        nargs="+",
        default=["clean", *ATTACKS, "IM"],
        choices=["clean", *ATTACKS, "IM"],
        help="Attacks to run; 'clean' is the unperturbed baseline, 'IM' is Induced Misdirection. Default: all.",
    )
    p.add_argument("--epsilon", type=float, default=0.0, help="IM: std of the Gaussian noise added to the misdirected pose.")
    p.add_argument(
        "--pose-pool-dir",
        type=Path,
        help="IM: draw poses from the .npy files in this folder (default: the other evaluated pairs).",
    )
    p.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    p.add_argument("--steps", type=int, default=10, help="DDIM steps.")
    p.add_argument("--seed", type=int, default=0, help="Base seed for perturbations and sampling noise.")
    p.add_argument("--ext", default="jpg", choices=["jpg", "png"], help="Output image format.")
    p.add_argument("--save-perturbed", action="store_true", help="Also save the perturbed source images.")
    p.add_argument("--skip-existing", action="store_true", help="Skip outputs that already exist.")
    args = p.parse_args()
    if args.epsilon < 0:
        p.error("--epsilon must be >= 0")
    return args


def load_model(pidm_root, config_path, ckpt_path, device):
    """Load the PIDM model, diffusion process and beta schedule."""
    sys.path.insert(0, str(pidm_root.resolve()))
    # Imported here because these modules live in the PIDM repository.
    from tensorfn import load_config
    from config.diffconfig import DiffusionConfig, get_model_conf
    from diffusion import create_gaussian_diffusion

    conf = load_config(DiffusionConfig, str(config_path), show=False)
    model = get_model_conf().make_model()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["ema"])
    model = model.to(device).eval()

    betas = conf.diffusion.beta_schedule.make()
    diffusion = create_gaussian_diffusion(betas, predict_xstart=False)
    return model, diffusion, betas


@torch.no_grad()
def predict_pose(src_bgr, tgt_pose, model, diffusion, betas, device, sampler="ddim", steps=100):
    """Generate the source person in the target pose.

    Args:
        src_bgr: uint8 BGR source image (OpenCV convention).
        tgt_pose: target-pose array of shape (C, H, W), see ``load_pose``.

    Returns:
        uint8 RGB image, HxWx3.
    """
    from diffusion import ddim_steps  # PIDM module, see load_model

    transform = T.Compose(
        [
            T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    src_pil = Image.fromarray(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB))
    src = transform(src_pil).unsqueeze(0).to(device)
    pose = torch.from_numpy(tgt_pose).unsqueeze(0).to(device)

    if sampler == "ddpm":
        samples = diffusion.p_sample_loop(model, x_cond=[src, pose], progress=True, cond_scale=2)
    else:
        noise = torch.randn(src.shape, device=device)
        seq = range(0, 1000, 1000 // steps)
        xs, _ = ddim_steps(noise, seq, model, betas.to(device), [src, pose])
        samples = xs[-1].to(device)

    out = (samples[0].clamp(-1.0, 1.0) + 1.0) / 2.0
    return (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA not available, running on CPU (very slow).")

    config = args.config or args.pidm_root / "config" / "diffusion.conf"
    ckpt = args.checkpoint or args.pidm_root / "checkpoints" / "last.pt"
    model, diffusion, betas = load_model(args.pidm_root, config, ckpt, device)

    # Samples that have both a source image and a target pose.
    samples = []
    for i in range(args.start, args.end + 1):
        src_path = args.input_dir / f"input{i}.jpg"
        pose_path = args.pose_dir / f"pose{i}.npy"
        if src_path.exists() and pose_path.exists():
            samples.append((i, src_path, pose_path))
        else:
            print(f"[{i}] missing {src_path.name} or {pose_path.name}, skipping")

    # Induced Misdirection: choose the pose handed to every sample up front.
    im_name = "IM" if args.epsilon == 0 else f"IM_eps{args.epsilon:g}"
    im_pose_path, im_bounds = {}, None
    if "IM" in args.attacks:
        eval_paths = [pose_path for _, _, pose_path in samples]
        pool = sorted(args.pose_pool_dir.glob("*.npy")) if args.pose_pool_dir else None
        if args.pose_pool_dir and not pool:
            sys.exit(f"No .npy files found in {args.pose_pool_dir}")
        rng = np.random.default_rng([args.seed, zlib.crc32(b"IM-choice")])
        chosen = choose_misdirection_poses(eval_paths, pool, rng)
        im_pose_path = {i: path for (i, _, _), path in zip(samples, chosen)}
        if args.epsilon > 0:
            im_bounds = pose_value_range(pool if pool else eval_paths)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output_dir / f"{im_name}_poses.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample", "misdirection_pose"])
            writer.writerows((i, path.name) for i, path in im_pose_path.items())

    for i, src_path, pose_path in samples:
        image = cv2.imread(str(src_path))
        if image is None:
            print(f"[{i}] cannot read {src_path}, skipping")
            continue
        pose = load_pose(pose_path)

        out_dir = args.output_dir / str(i)
        out_dir.mkdir(parents=True, exist_ok=True)

        for name in args.attacks:
            stem = "B1" if name == "clean" else f"B1_{im_name if name == 'IM' else name}"
            out_path = out_dir / f"{stem}.{args.ext}"
            if args.skip_existing and out_path.exists():
                continue

            # Seed per (sample, attack) so results do not depend on run order
            # or on which other attacks were selected.
            rng = np.random.default_rng([args.seed, i, zlib.crc32(name.encode())])
            src, tgt = image, pose
            if name == "IM":  # pose attack: the source image stays untouched
                tgt = add_pose_noise(load_pose(im_pose_path[i]), args.epsilon, im_bounds, rng)
            elif name != "clean":
                src = ATTACKS[name](image.copy(), rng=rng)
                if args.save_perturbed:
                    cv2.imwrite(str(out_dir / f"input_{name}.{args.ext}"), src)

            # Same sampling noise for every attack on a sample, so differences
            # between outputs come from the perturbation alone.
            torch.manual_seed(args.seed + i)
            result = predict_pose(src, tgt, model, diffusion, betas, device, args.sampler, args.steps)
            Image.fromarray(result).save(out_path)

        print(f"[{i}/{args.end}] done")

    print("All iterations completed.")


if __name__ == "__main__":
    main()
