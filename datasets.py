"""Custom datasets for DARTS architecture search.

Based on NpyWebDataset pattern: datasets stored as .npy files in zip archives
downloaded from remote URLs.
"""
import hashlib
import os
import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests
import torch


class NpyWebDataset(torch.utils.data.Dataset):
    """Dataset backed by .npy files downloaded from a URL and extracted from a zip archive.

    Data is expected to be organized as pairs of _x (data) and _y (label) .npy files
    with filenames containing 'train'/'valid' or 'test' prefixes.
    """

    def __init__(
        self,
        url: str,
        train: bool = True,
        root: str = "data/webdatasets/npy",
        name: str = "",
        download: bool = True,
        transform: Optional[Callable] = None,
        data_key: str = "_x",
        label_key: str = "_y",
    ):
        self.url = url
        self.name = name
        self.train = train
        self.root = Path(os.path.expanduser(root))
        self.download = download
        self.transform = transform
        self.data_key = data_key
        self.label_key = label_key

        self.local_zip_path = self._download_and_extract()
        self.data_files, self.label_files = self._find_data_and_labels()
        self.data, self.labels = self._load_data() if download else (None, None)

    @property
    def targets(self):
        return self.labels

    def _download_and_extract(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        hash_name = self.name if self.name else hashlib.sha256(self.url.encode()).hexdigest()
        zip_path = self.root / f"{hash_name}.zip"
        extract_dir = self.root / f"{hash_name}_extracted"

        if not zip_path.exists():
            r = requests.get(self.url)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(r.content)

        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

        return extract_dir

    def _find_data_and_labels(self) -> Tuple[List[Path], List[Path]]:
        files = list(self.local_zip_path.rglob("*.npy"))
        prefix = ["train", "valid"] if self.train else ["test"]

        data_files = sorted(
            f for f in files
            if (self.data_key in f.name) and any(ix in f.name for ix in prefix)
        )
        label_files = sorted(
            f for f in files
            if (self.label_key in f.name) and any(ix in f.name for ix in prefix)
        )

        assert len(data_files) == len(label_files), "Mismatch in data/label file counts"
        assert len(data_files) > 0, f"No matching .npy files found for prefix '{prefix}'"

        return data_files, label_files

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        data = np.concatenate([np.load(f) for f in self.data_files], axis=0)
        labels = np.concatenate([np.load(f) for f in self.label_files], axis=0)
        return data, labels.astype(np.int64)

    def __getitem__(self, index):
        if self.download:
            x, y = self.data[index], self.labels[index]
        else:
            file_idx, local_idx = self._resolve_index(index)
            x = np.load(self.data_files[file_idx])[local_idx]
            y = np.load(self.label_files[file_idx])[local_idx]

        if self.transform is not None:
            x = self.transform(x)
        return x, int(y)

    def _resolve_index(self, index) -> Tuple[int, int]:
        cumulative = 0
        for i, file in enumerate(self.data_files):
            n = np.load(file, mmap_mode="r").shape[0]
            if index < cumulative + n:
                return i, index - cumulative
            cumulative += n
        raise IndexError("Index out of bounds")

    def __len__(self):
        if self.download:
            return len(self.data)
        return sum(np.load(f, mmap_mode="r").shape[0] for f in self.data_files)


class AddNIST(NpyWebDataset):
    def __init__(self, train=True, root="data/webdatasets/npy", download=True, transform=None):
        super().__init__(
            url="https://data.ncl.ac.uk/ndownloader/articles/24574354/versions/1",
            name="AddNIST", train=train, root=root, download=download, transform=transform,
        )
        self.data = self.data.transpose(0, 2, 3, 1)


class MultNIST(NpyWebDataset):
    def __init__(self, train=True, root="data/webdatasets/npy", download=True, transform=None):
        super().__init__(
            url="https://data.ncl.ac.uk/ndownloader/articles/24574678/versions/1",
            name="MultNIST", train=train, root=root, download=download, transform=transform,
        )
        self.data = self.data.transpose(0, 2, 3, 1)


class CIFARTile(NpyWebDataset):
    def __init__(self, train=True, root="data/webdatasets/npy", download=True, transform=None):
        super().__init__(
            url="https://data.ncl.ac.uk/ndownloader/articles/24551539/versions/1",
            name="CIFARTile", train=train, root=root, download=download, transform=transform,
        )
        self.data = self.data.transpose(0, 2, 3, 1)


class LanguageASPELL(NpyWebDataset):
    def __init__(self, train=True, root="data/webdatasets/npy", download=True, transform=None):
        super().__init__(
            url="https://data.ncl.ac.uk/ndownloader/articles/24574729/versions/1",
            name="LanguageASPELL", train=train, root=root, download=download, transform=transform,
        )
        self.data = self.data.transpose(0, 2, 3, 1)


class Gutenberg(NpyWebDataset):
    def __init__(self, train=True, root="data/webdatasets/npy", download=True, transform=None):
        super().__init__(
            url="https://data.ncl.ac.uk/ndownloader/articles/24574753/versions/1",
            name="Gutenberg", train=train, root=root, download=download, transform=transform,
        )
        self.data = self.data.transpose(0, 2, 3, 1)
        self.data = np.pad(self.data, ((0, 0), (0, 1), (1, 1), (0, 0)))  # H: 27→28, W: 18→20


class GeoClassing(NpyWebDataset):
    def __init__(self, train=True, root="data/webdatasets/npy", download=True, transform=None):
        super().__init__(
            url="https://data.ncl.ac.uk/ndownloader/articles/24050256/versions/3",
            name="GeoClassing", train=train, root=root, download=download, transform=transform,
        )
        self.data = self.data.transpose(0, 2, 3, 1)


class Chesseract(NpyWebDataset):
    def __init__(self, train=True, root="data/webdatasets/npy", download=True, transform=None):
        super().__init__(
            url="https://data.ncl.ac.uk/ndownloader/articles/24118743/versions/2",
            name="Chesseract", train=train, root=root, download=download, transform=transform,
        )
        self.data = self.data.transpose(0, 2, 3, 1)


class GameOfLife(NpyWebDataset):
    def __init__(self, train=True, root="data/webdatasets/npy", download=True, transform=None):
        super().__init__(
            url="https://data.ncl.ac.uk/ndownloader/articles/30000835/versions/1",
            name="GameOfLife", train=train, root=root, download=download, transform=transform,
        )
        self.data = self.data[:, :, :, None]


# Registry of all custom datasets (lowercase name -> class)
CUSTOM_DATASETS = {
    'addnist': AddNIST,
    'multnist': MultNIST,
    'cifartile': CIFARTile,
    'language': LanguageASPELL,
    'gutenberg': Gutenberg,
    'geoclassing': GeoClassing,
    'chesseract': Chesseract,
    'gameoflife': GameOfLife,
}

# Number of output classes per custom dataset
N_CLASSES = {
    'addnist': 20,
    'multnist': 10,
    'cifartile': 4,
    'language': 10,
    'gutenberg': 6,
    'geoclassing': 10,
    'chesseract': 3,
    'gameoflife': 25,
}
