from ..MD.paths import MD_OUT_DIR


def complex_dir(pdb_id):
    return MD_OUT_DIR / pdb_id


def features_dir(pdb_id):
    return complex_dir(pdb_id) / "features"
