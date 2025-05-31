# src/data/load_data.py

import pandas as pd
from pathlib import Path

# Directorio donde deben ir los CSV en crudo
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def load_raw_pokemon(filename: str = "pokemon.csv") -> pd.DataFrame:
    """
    Carga el CSV de Pokémon desde data/raw y devuelve un DataFrame.
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"No encontré {path}. Asegúrate de poner el CSV allí.")
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    # Prueba rápida desde línea de comando o IDE
    df = load_raw_pokemon()
    print("Shape:", df.shape)
    print(df.head())
