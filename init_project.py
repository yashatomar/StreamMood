import os

directories = [
    "api",
    "app",
    "data/raw",
    "data/processed",
    "models",
    "pipeline",
    "notebooks",
    "tests"
]

for folder in directories:
    os.makedirs(folder, exist_ok=True)
    print(f"[CREATED] {folder}/")

print("\nProject structure is ready!")