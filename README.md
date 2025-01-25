AttentionSDF
==============================

Attention based model for continuous SDF(signed distance function) prediction inspired by DeepSDF paper.

## Installation

To set up the environment and install dependencies, follow these steps:

```bash
python -m venv venv
source ./venv/Scripts/activate  # Use `source ./venv/bin/activate` on macOS/Linux
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Running Experiments

### Preparing Training Data

1. Place the training data in `.obj` format in the `data/raw` directory.
2. Normalize and preprocess the models by running:

   ```bash
   python -m src.data.normalize_models
   python -m src.data.process_models
   ```

   This will generate:
   - Normalized `.obj` files in the `data/intermediate` folder.
   - Processed `.hdf5` files in the `data/processed` folder, which will be used during training.

### Training the Model

1. The model definition can be found and modified in `src/models/model.py`.
2. Training parameters are located in `src/models/train.py`.
3. To start training, run:

   ```bash
   python -m src.models.train
   ```

   During training:
   - Results will be saved in the `results/[Year]-[Month]-[Day]-[Hour]-[Minute]-[Second]` directory, where the timestamp represents the training start time.
   - The initial configuration is saved as `config.json` in the results folder.
   - Model weights and biases are saved in the `results` folder, configurable via `TrainingArguments`.
   - The final model is saved in a folder named `[Year]-[Month]-[Day]`.

   For each curriculum stage, new models will be generated for each `.obj` file in `data/intermediate` with the naming convention:

   ```
   [model_name]-[timestamp]-curriculum-[stage].obj
   ```

### Generating New Meshes

To generate new `.obj` files using a trained model:

1. Configure the `config.json` file and use the corresponding `model.safetensors` file.
2. Run the mesh generation script:

   ```bash
   python -m src.models.generate_mesh
   ```

   Results will be saved in the `generated_meshes` folder within the experiment's results directory.

####

## Project Organization

    ├── LICENSE             <- N/A license file
    ├── README.md           <- The top-level README for individuals using this project
    ├── data
    │   ├── intermediate    <- Intermediate data that has been transformed
    │   ├── processed       <- The final, canonical data sets for modeling
    │   └── raw             <- The original, immutable data dump
    │
    ├── notebooks           <- Jupyter notebooks. Containg experiments.
    |
    ├── results             <- Generated results from fitting models
    │
    ├── src                 <- Source code for use in this project
    │   ├── __init__.py     <- Makes src a Python module
    │   │
    │   ├── data            <- Scripts to load and process data
    │   │   └── load_data.py
    |   |   └── normalize_models.py
    │   │   └── process_models.py
    │   │   └── lazy_sample_loader.py
    │   │
    │   ├── models          <- Scripts for models and fitting processed data
    │   │   └── dataset.py
    │   │   └── loss.py
    │   │   └── modules.py
    │   │   └── model.py
    │   │   └── train.py
    │   │   └── metrics.py
    │   │   └── generate_mesh.py
    │   │
    │   └── visualization   <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize_point_cloud.py
    │       └── generate_mesh.py
    │
    ├── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g.
    │                          generated with `pip freeze > requirements.txt`
    │
    └── setup.py            <- makes project pip installable (pip install -e .) so src can be imported


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
