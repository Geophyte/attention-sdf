""" Module for sampling SDF values from normalized model using method defined in
    DeepSDF paper
"""

import numpy as np
from pathlib import Path
import trimesh
from tqdm import tqdm
from mesh_to_sdf import sample_sdf_near_surface
from src.data.load_data import get_data_dir
from src.data.lazy_sample_loader import LazySampleLoader


np.random.seed(42)


def merge_meshes(meshes: list) -> trimesh.Trimesh:
    merged_mesh = trimesh.util.concatenate(meshes)
    return merged_mesh


def sample_sdf_values(model_path: Path, num_samples: int) -> np.ndarray:
    mesh = trimesh.load_mesh(model_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = merge_meshes([geometry for geometry in mesh.geometry.values()])
    points, sdf = sample_sdf_near_surface(mesh, number_of_points=num_samples)
    samples = np.hstack((points, sdf.reshape(-1, 1)))
    return samples


def batch(lst: list, n=1):
    l = len(lst)
    for ndx in range(0, l, n):
        yield lst[ndx:min(ndx + n, l)]


def save_samples(batches: np.ndarray, context_size: int, input_path: Path,
    target_dir: Path, source_dir: Path, suffix: str):
    loader = LazySampleLoader(batches, context_size)
    relative_path = input_path.relative_to(source_dir)
    output_path = target_dir / relative_path.with_name(f"{relative_path.stem}_{suffix}.hdf5")
    loader.save_to_file(output_path)
    loader.close()
    pass


def split_samples(samples: np.ndarray, batch_size: int, train_batches: int, num_batches: int):
    batches = samples[:num_batches * batch_size].reshape(num_batches, batch_size, 4)
    training_batches = batches[:train_batches]
    validation_batches = batches[train_batches:]
    return training_batches, validation_batches


def process_models(source_dir: Path, target_dir: Path, context_size: int,
    sample_size: int, train_batches: int, val_batches: int) -> None:
    paths = list(source_dir.rglob('*.obj'))
    batch_size = context_size + sample_size
    num_batches = train_batches + val_batches
    num_samples = batch_size * num_batches

    with tqdm(total=len(paths), desc="Processing files") as pbar:
        for model_file in paths:
            samples = sample_sdf_values(model_file, num_samples)
            np.random.shuffle(samples)
            training_batches, validation_batches = split_samples(samples, batch_size, train_batches, num_batches)
            save_samples(training_batches, context_size, model_file, target_dir, source_dir, suffix="train")
            save_samples(validation_batches, context_size, model_file, target_dir, source_dir, suffix="val")
            pbar.update(1)


if __name__ == '__main__':
    data_dir = get_data_dir()

    source_dir = data_dir / 'intermediate'
    target_dir = data_dir / 'processed'

    process_models(source_dir, target_dir, context_size=200, sample_size=20, train_batches=18_000, val_batches=200)
