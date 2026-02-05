from pandas import read_csv
from .paths import DATA_DIR

def load_prodigy_data(csv_name="PRODIGY_Data.csv"):
    # Expected at src/bridging/processedData/PRODIGY_Data.csv
    return read_csv(DATA_DIR / csv_name)

def parse_complex_pdb(complex_pdb):
    """Parse strings like '1A2K_C:AB' -> ('1A2K', ['C','A','B'])."""
    pdb_id, chains_part = complex_pdb.split("_", 1)
    left, right = chains_part.split(":")
    chain_ids = list(dict.fromkeys(list(left) + list(right)))  # preserve order, unique
    return pdb_id, chain_ids
