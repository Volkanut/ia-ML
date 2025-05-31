# src/data/download_data.py
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path

def download_pokemon(dest: str = None):
    """
    Descarga y descomprime el dataset de Pokémon desde Kaggle
    al directorio data/raw.
    """
    api = KaggleApi()
    api.authenticate()

    dest = Path(dest or Path(__file__).resolve().parents[2] / "data" / "raw")
    dest.mkdir(parents=True, exist_ok=True)

    api.dataset_download_files(
        "abcsds/pokemon",
        path=str(dest),
        unzip=True,
        quiet=False
    )
    print("Descarga completa en:", dest)

if __name__ == "__main__":
    download_pokemon()
