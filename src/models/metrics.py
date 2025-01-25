import torch
import trimesh
import numpy as np
from chamferdist import ChamferDistance
from scipy.stats import wasserstein_distance_nd
from pathlib import Path
from src.data.load_data import get_results_dir, get_data_dir


def sample_points_on_mesh(mesh: trimesh.Trimesh, num_samples: int) -> torch.Tensor:
    points = mesh.sample(num_samples)
    return torch.tensor(points, dtype=torch.float32).unsqueeze(0)


def chamfer_distance(points1: torch.Tensor, points2: torch.Tensor) -> float:
    chamfer = ChamferDistance()
    dist = chamfer(points1, points2)
    return dist.item()


def _reshape(points: torch.Tensor) -> np.array:
    B, N, M = points.shape
    return points.reshape(B * N, M).numpy()


def earth_movers_distance(points1: torch.Tensor, points2: torch.Tensor) -> float:
    points1_reshaped = _reshape(points1)
    points2_reshaped = _reshape(points2)
    return wasserstein_distance_nd(points1_reshaped, points2_reshaped)


if __name__ == "__main__":
    reconstructed_dir = (
        get_results_dir() / "2025_01_22_no_curriculum-2025-01-22-18-08-59"
    )
    original_dir = get_data_dir() / "intermediate"

    reconstruction_original_pairs = [
        ("bottle-2025-01-22-18-08-59-curriculum-0.obj", "bottle/bottle.obj"),
        ("camera-2025-01-22-18-08-59-curriculum-0.obj", "camera/camera.obj"),
        ("cup-2025-01-22-18-08-59-curriculum-0.obj", "cup/cup.obj"),
    ]

    for reconstruction_path, original_path in reconstruction_original_pairs:
        reconstruction = trimesh.load_mesh(reconstructed_dir / reconstruction_path)
        original = trimesh.load_mesh(original_dir / original_path)

        points1_cd = sample_points_on_mesh(reconstruction, 30_000)
        points2_cd = sample_points_on_mesh(original, 30_000)

        print(f"{reconstruction_path}:")

        chamfer_dist = chamfer_distance(points1_cd, points2_cd)
        print(f"    Chamfer Distance: {chamfer_dist}")

        points1_emd = sample_points_on_mesh(reconstruction, 500)
        points2_emd = sample_points_on_mesh(original, 500)

        emd = earth_movers_distance(points1_emd, points2_emd)
        print(f"    Earth Mover's Distance: {emd}")
