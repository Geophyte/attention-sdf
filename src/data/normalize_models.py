"""
Module for normalizing meshes into bounding box defined by p1(1, 1, 1) and
p2(-1, -1, -1). If run normalizes recursivly raw .objs into intermediate .objs
"""

import trimesh
from pathlib import Path
from tqdm import tqdm
from src.data.load_data import get_data_dir


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.apply_translation(-mesh.centroid)
    scale_factor = 1.0 / mesh.scale
    mesh.apply_scale(scale_factor)
    return mesh


def normalize_obj_file(input_path: Path, target_dir: Path, source_dir: Path) -> None:
    mesh = trimesh.load(input_path)
    mesh = normalize_mesh(mesh)

    relative_path = input_path.relative_to(source_dir)
    output_path = target_dir / relative_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)


def normalize_obj_files(source_dir: Path, target_dir: Path) -> None:
    paths = list(source_dir.rglob("*.obj"))
    for obj_file in tqdm(paths, desc="Normalizing OBJs"):
        normalize_obj_file(obj_file, target_dir, source_dir)


if __name__ == "__main__":
    data_dir = get_data_dir()

    source_dir = data_dir / "raw"
    target_dir = data_dir / "intermediate"

    normalize_obj_files(source_dir, target_dir)
