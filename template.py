import os
from pathlib import Path

list_of_files = [

    # Data folders
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",

    # Models
    "models/sign_model.h5",

    # Notebooks
    "notebooks/01_data_collection.ipynb",
    "notebooks/02_model_training.ipynb",
    "notebooks/03_evaluation.ipynb",

    # Source code
    "src/__init__.py",
    "src/data_collection.py",
    "src/model.py",
    "src/predict.py",
    "src/tts.py",

    # App
    "app.py",

    # Config files
    "requirements.txt",
    "README.md",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            pass

print("✅ Project structure created successfully!")