import json
from pathlib import Path

import torch
import safetensors.torch as safetensors
from src.models.model import SDFTransformer, SDFTransformerConfig
from src.visualization.generate_mesh import generate_meshes
from src.data.load_data import get_data_dir, get_results_dir

BATCH_SIZE = 64
RESOLUTION = 256
CONTEXT_SIZE = 256


def load_model(model_dir: Path) -> SDFTransformer:
    """Load a saved model and its configuration."""
    config_path = next(model_dir.glob("*config.json"), None)
    if not config_path:
        raise FileNotFoundError(
            f"No config file found in {model_dir} matching pattern *config.json"
        )

    model_path = next(model_dir.glob("*model"), None)
    if not model_path:
        raise FileNotFoundError(
            f"No model folder found in {model_dir} matching pattern *model"
        )
    model_path = model_path / "model.safetensors"

    with open(config_path, "r") as f:
        config_data = json.load(f)
    config = SDFTransformerConfig(**config_data)

    model = SDFTransformer(config)
    safetensors.load_model(model, model_path)
    model.eval()

    return model


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = get_results_dir() / "2025-01-25-14-55-10"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

    model = load_model(model_dir).to(device)

    obj_dir = get_data_dir() / "intermediate"
    result_dir = model_dir / "generated_meshes"
    result_dir.mkdir(parents=True, exist_ok=True)

    format_string = f"resolution-{RESOLUTION}.obj"
    generate_meshes(
        model=model,
        obj_dir=obj_dir,
        output_dir=result_dir,
        naming_format=format_string,
        device=device,
        batch_size=BATCH_SIZE,
        resolution=RESOLUTION,
        context_size=CONTEXT_SIZE,
    )
