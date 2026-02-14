import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import FEATURES_FILENAME


def collect_feature_files(path_or_glob, filename=FEATURES_FILENAME):
    path = Path(path_or_glob)
    if path.exists() and path.is_dir():
        return sorted([str(p) for p in path.rglob(filename)])
    return sorted(glob.glob(str(path_or_glob)))


def feature_pdb_id(path_like: str) -> str | None:
    path = Path(path_like)
    if path.parent.name != "features":
        return None
    pdb = path.parent.parent.name.strip().upper()
    if len(pdb) != 4:
        return None
    return pdb


class FeatureFrameDataset(Dataset):
    """
    Dataset over individual frames from feature arrays shaped (T, C, N, N).
    Uses memory mapping to avoid loading everything at once.
    """

    def __init__(self, npy_paths, frame_stride=1, max_frames=None, target_by_pdb=None):
        if not npy_paths:
            raise ValueError("No feature files provided.")
        self.npy_paths = list(npy_paths)
        self.frame_stride = max(1, int(frame_stride))
        self.max_frames = max_frames
        self.target_by_pdb = target_by_pdb if target_by_pdb else None

        self._arrays = [None] * len(self.npy_paths)
        self._index = []

        for i, path in enumerate(self.npy_paths):
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 4:
                raise ValueError(f"{path} expected shape (T,C,N,N), got {arr.shape}")
            total = arr.shape[0]
            frames = list(range(0, total, self.frame_stride))
            if self.max_frames is not None:
                frames = frames[: int(self.max_frames)]
            pdb = feature_pdb_id(path)
            y = None
            if self.target_by_pdb is not None and pdb is not None:
                y = self.target_by_pdb.get(pdb)
            for t in frames:
                self._index.append((i, t, y))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        file_i, t, y = self._index[idx]
        if self._arrays[file_i] is None:
            self._arrays[file_i] = np.load(self.npy_paths[file_i], mmap_mode="r")
        x = self._arrays[file_i][t].astype(np.float32)
        x_t = torch.from_numpy(x)
        if self.target_by_pdb is None:
            return x_t
        if y is None:
            return x_t, torch.tensor(0.0, dtype=torch.float32), torch.tensor(False)
        return x_t, torch.tensor(float(y), dtype=torch.float32), torch.tensor(True)
