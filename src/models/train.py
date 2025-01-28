import json
from datetime import datetime
from dataclasses import asdict

import torch
from transformers import Trainer, TrainingArguments

from src.data.load_data import get_results_dir, get_data_dir
from src.models.dataset import LazySampleDataset
from src.models.model import SDFTransformer, SDFTransformerConfig
from src.visualization.generate_mesh import generate_meshes


BATCH_SIZE = 64
CURRICULUM_SCHEDULE = [
    {
        "epochs": 2,
        "epsilon": 0.02,
        "lambda": 0.0,
        "learning_rate": 5e-5,
        "resolution": 100,
    },
    {
        "epochs": 2,
        "epsilon": 0.0075,
        "lambda": 0.15,
        "learning_rate": 4e-5,
        "resolution": 100,
    },
    {
        "epochs": 2,
        "epsilon": 0.004,
        "lambda": 0.3,
        "learning_rate": 3e-5,
        "resolution": 100,
    },
    {
        "epochs": 2,
        "epsilon": 0.002,
        "lambda": 0.4,
        "learning_rate": 2e-5,
        "resolution": 100,
    },
    {
        "epochs": 2,
        "epsilon": 0.0,
        "lambda": 0.5,
        "learning_rate": 1e-5,
        "resolution": 256,
    },
]

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = SDFTransformerConfig()
    model = SDFTransformer(config).to(device)

    processed_dir = get_data_dir() / "processed"
    train_files = list(processed_dir.rglob("*_train.hdf5"))
    val_files = list(processed_dir.rglob("*_val.hdf5"))

    train_dataset = LazySampleDataset(train_files)
    val_dataset = LazySampleDataset(val_files)

    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    result_dir = get_results_dir() / current_date
    result_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=result_dir / "results",
        eval_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=1,
        logging_dir=result_dir / "logs",
        logging_steps=10,
        weight_decay=0.01,
        save_total_limit=3,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    obj_dir = get_data_dir() / "intermediate"
    format_string_base = f"{{name}}-{current_date}-curriculum-"

    config_name = f"config.json"
    with open(result_dir / config_name, "w") as f:
            json.dump(asdict(config), f)

    for i, stage in enumerate(CURRICULUM_SCHEDULE):
        model.epsilon = stage["epsilon"]
        model.lambdaa = stage["lambda"]
        trainer.args.num_train_epochs = stage["epochs"]
        trainer.args.learning_rate = stage["learning_rate"]

        trainer.train()

        if stage["resolution"] is not None:
            format_string = format_string_base + str(i) + ".obj"
            generate_meshes(
                model=model,
                obj_dir=obj_dir,
                output_dir=result_dir,
                naming_format=format_string,
                device=device,
                batch_size=BATCH_SIZE,
                resolution=stage["resolution"],
                context_size=512,
            )

    train_dataset.close()
    val_dataset.close()

    current_date = datetime.now().strftime("%Y-%m-%d")
    model_name = f"{current_date}-model"
    trainer.save_model(result_dir / model_name)
