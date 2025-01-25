from pathlib import Path
import torch
from torch.utils.data import Dataset
from src.data.lazy_sample_loader import LazySampleLoader
import numpy as np


class LazySampleDataset(Dataset):
    def __init__(self, filepaths: list[str]) -> None:
        self.filepaths = [Path(filepath) for filepath in filepaths]
        self.loaders = [
            LazySampleLoader(filepath=filepath) for filepath in self.filepaths
        ]
        self.total_samples = 0
        self.sample_indices_map = []

        for loader in self.loaders:
            num_batches, num_samples_per_batch = loader.get_batch_and_sample_count()
            self.total_samples += num_batches * num_samples_per_batch
            self.sample_indices_map.extend(
                [
                    (loader, batch_index, sample_index)
                    for batch_index in range(num_batches)
                    for sample_index in range(num_samples_per_batch)
                ]
            )

    def _sample_indices_map(self, index: int) -> tuple[LazySampleLoader, int, int]:
        if index < 0 or index >= self.total_samples:
            raise IndexError(
                f"Index {index} out of bounds for dataset of size {self.total_samples}"
            )
        return self.sample_indices_map[index]

    def __getitem__(self, index: int) -> dict[str, any]:
        loader, batch_index, sample_index = self._sample_indices_map(index)
        # context shape: [num_samples, 4]
        # input_data shape: [1, 3]
        # label_data shape: 1
        context, input_data, label_data = loader.get_sample(batch_index, sample_index)

        # input_data shape: [1, 3]
        # label_data shape: [1, 1]
        input_data = input_data.reshape(1, 3)
        label_data = np.full((1, 1), label_data, dtype=np.float32)

        context_tensor = torch.from_numpy(context)
        input_tensor = torch.from_numpy(input_data)
        label_tensor = torch.from_numpy(label_data)

        return {"context": context_tensor, "x": input_tensor, "labels": label_tensor}

    def __len__(self) -> int:
        return self.total_samples

    def close(self):
        for loader in self.loaders:
            loader.close()
