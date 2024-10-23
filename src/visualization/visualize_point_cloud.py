import numpy as np
from typing import Tuple
from pathlib import Path
import pyrender
from src.data.load_data import get_data_dir
from src.data.lazy_sample_loader import LazySampleLoader

PointData = Tuple[float, float, float, float]

def visualize_point_cloud(file_path: Path) -> None:
    loader = LazySampleLoader(filepath=file_path)
    samples = loader.get_all_contexts()
    points = samples[:, :3]
    sdf = samples[:, 3]
    colors = np.zeros(points.shape)
    colors[sdf < 0, 2] = 1
    colors[sdf > 0, 0] = 1

    cloud = pyrender.Mesh.from_points(points, colors=colors)
    scene = pyrender.Scene()
    scene.add(cloud)
    pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

if __name__ == "__main__":
    data_dir = get_data_dir()
    sample_dir = data_dir / 'processed' / 'bunny' / 'stanford-bunny.hdf5'
    visualize_point_cloud(sample_dir)