"""
Module with helper class for creating and loading sdf sample structures for
training and visualization
"""

import h5py
import numpy as np
from pathlib import Path


class LazySampleLoader:
    """Class must be either used for saving or loading. Same instance can't be
    used for both
    """

    def __init__(
        self,
        batches: np.ndarray = None,
        context_size: int = None,
        filepath: Path = None,
    ) -> None:
        self.context_size = context_size
        self.filepath = filepath
        self.file = None

        if batches is not None and context_size is not None:
            self.contexts = []
            self.inputs = []
            self.labels = []

            for batch in batches:
                context = batch[:context_size]
                remainder = batch[context_size:]

                input_data = remainder[:, :3]
                label_data = remainder[:, 3]

                self.contexts.append(context)
                self.inputs.append(input_data)
                self.labels.append(label_data)

        elif filepath:
            self.load_from_file(filepath)

    def save_to_file(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(filepath, "w") as f:
            for i, (context, input_data, label_data) in enumerate(
                zip(self.contexts, self.inputs, self.labels)
            ):
                f.create_dataset(f"batch_{i}/context", data=context)
                f.create_dataset(f"batch_{i}/input", data=input_data)
                f.create_dataset(f"batch_{i}/labels", data=label_data)

    def load_from_file(self, filepath: Path):
        self.filepath = filepath
        self.file = h5py.File(filepath, "r")
        self.num_batches = len(self.file.keys())

    def get_batch(self, batch_index) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        context = self.file[f"batch_{batch_index}/context"][:]
        input_data = self.file[f"batch_{batch_index}/input"][:]
        label_data = self.file[f"batch_{batch_index}/labels"][:]
        return context, input_data, label_data

    def get_sample(
        self, batch_index, sample_index
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        context = self.file[f"batch_{batch_index}/context"][:]
        input_data = self.file[f"batch_{batch_index}/input"][sample_index]
        label_data = self.file[f"batch_{batch_index}/labels"][sample_index]
        return context, input_data, label_data

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def get_all_inputs_with_labels(self):
        all_inputs = []
        all_labels = []

        for batch_index in range(self.num_batches):
            inputs = self.file[f"batch_{batch_index}/input"][:]
            labels = self.file[f"batch_{batch_index}/labels"][:]

            all_inputs.append(inputs)
            all_labels.append(labels)

        all_inputs = np.concatenate(all_inputs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return np.hstack((all_inputs, all_labels.reshape(-1, 1)))

    def get_all_contexts(self):
        all_contexts = []

        for batch_index in range(self.num_batches):
            context = self.file[f"batch_{batch_index}/context"][:]
            all_contexts.append(context)
        return np.concatenate(all_contexts, axis=0)

    def get_batch_and_sample_count(self):
        num_batches = self.num_batches
        num_samples_per_batch = self.file[f"batch_{0}/input"].shape[0]
        return num_batches, num_samples_per_batch
