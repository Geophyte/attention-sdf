import trimesh
import numpy as np
from skimage import measure
from typing import Callable, Tuple, List
from pathlib import Path
import torch
from src.data.process_models import sample_sdf_values
from src.data.lazy_sample_loader import LazySampleLoader

try:
    from IPython import get_ipython

    in_notebook = get_ipython() is not None
except ImportError:
    in_notebook = False
if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def _sample_sdf(
    sdf_func: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    resolution: int,
    batch_size: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(bounds[0][0], bounds[0][1], resolution)
    y = np.linspace(bounds[1][0], bounds[1][1], resolution)
    z = np.linspace(bounds[2][0], bounds[2][1], resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    positions = np.stack([X, Y, Z], axis=-1)
    n = positions.shape[0]
    positions = positions.reshape(-1, 3)
    sdf_values = []
    for i in tqdm(range(0, positions.shape[0], batch_size), desc="Processing batches"):
        batch = positions[i : i + batch_size]
        sdf_values.append(sdf_func(batch.reshape(-1, 3)))
    sdf_values = np.concatenate(sdf_values)
    sdf_values = sdf_values.reshape((n, n, n))
    return sdf_values, x, y, z


def _calculate_spacing(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[float, float, float]:
    return x[1] - x[0], y[1] - y[0], z[1] - z[0]


def _get_verts_faces(
    sdf_values: np.ndarray, spacing: Tuple[float, float, float], level: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    verts, faces, normals, values = measure.marching_cubes(
        sdf_values, level, spacing=spacing
    )
    return verts, faces


def _save_mesh_to_obj(filename: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """
    Saves a mesh to an .obj file.

    Args:
        filename (str): The name of the file to save the mesh to.
        verts (np.ndarray): The vertices of the mesh.
        faces (np.ndarray): The faces of the mesh.
    """
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(filename)


def _generate_mesh(
    sdf_func: Callable[[np.ndarray], float],
    bounds: List[Tuple[float, float]],
    batch_size: int,
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a mesh from an SDF function.

    Args:
        sdf_func (Callable[[np.ndarray], float]): The SDF function to generate the mesh from.
        bounds (List[Tuple[float, float]]): The bounds for the 3D grid.
        resolution (int): The resolution of the 3D grid.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The vertices and faces of the generated mesh.
    """
    sdf_values, x, y, z = _sample_sdf(sdf_func, bounds, resolution, batch_size)
    spacing = _calculate_spacing(x, y, z)
    verts, faces = _get_verts_faces(sdf_values, spacing)
    return verts, faces


def _get_model_prediction(
    model,
    context: np.ndarray,
    input_data: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).to(device)
    context_tensor = context_tensor.repeat(batch_size, 1, 1)
    input_data_tensor = (
        torch.tensor(input_data, dtype=torch.float32).unsqueeze(1).to(device)
    )

    output: torch.Tensor = model.forward(context_tensor, input_data_tensor)["logits"]
    return output.squeeze().cpu().numpy()


def generate_mesh(
    model,
    file_path: Path,
    output_path: Path,
    device: torch.device,
    batch_size: int,
    bounds: list[tuple] = [(-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5)],
    resolution: int = 200,
    context_size: int = 50,
):

    if file_path.suffix == ".obj":
        context = sample_sdf_values(file_path, context_size)
    elif file_path.suffix == ".hdf5":
        loader = LazySampleLoader(filepath=file_path)
        context, _, _ = loader.get_batch(0)
        loader.close()
    sdf_func = lambda p: _get_model_prediction(
        model, context, p, device, batch_size=batch_size
    )
    with torch.no_grad():
        verts, faces = _generate_mesh(
            sdf_func, bounds, batch_size=batch_size, resolution=resolution
        )
    _save_mesh_to_obj(output_path, verts, faces)


def generate_meshes(
    model,
    obj_dir: Path,
    output_dir: Path,
    naming_format: str,
    device: torch.device,
    batch_size: int,
    resolution: int = 100,
    context_size: int = 50,
    bounds: list[tuple] = [(-1.2, 1.2), (-1.2, 1.2), (-1.2, 1.2)],
):
    obj_files = list(obj_dir.rglob("*.obj"))
    for obj_path in tqdm(obj_files, desc="Processing models"):
        relative_path = obj_path.relative_to(obj_dir)
        file_stem = relative_path.stem
        mesh_name = naming_format.format(name=file_stem)
        mesh_output_path = output_dir / mesh_name
        generate_mesh(
            model,
            obj_path,
            mesh_output_path,
            device,
            batch_size,
            bounds=bounds,
            resolution=resolution,
            context_size=context_size,
        )
