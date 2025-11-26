# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from typing import Optional, Dict, Any

from generic_neuromotor_interface.data import (
    DataSplit,
    make_dataset,
    make_handwriting_dataset,
)
from generic_neuromotor_interface.transforms import HandwritingTransform, Transform
from generic_neuromotor_interface.utils import handwriting_collate
from torch.utils.data import DataLoader, default_collate


# channel-augmentation helpers for collate-time
def _sample_rotation_k(cfg: Dict[str, Any] | None, C: int, g: Optional[torch.Generator]) -> int:
    if not cfg or not cfg.get("enable", False):
        return 0
    if cfg.get("random_k", False):
        lo, hi = cfg.get("k_range", [-8, 8])
        k = int(torch.randint(low=int(lo), high=int(hi)+1, size=(1,), generator=g).item())
    else:
        k = int(cfg.get("k", 0))
    # normalize to shortest equivalent rotation
    k = ((k % C) + C) % C
    if k > C // 2:
        k = k - C
    return k

def _apply_rotation(x: torch.Tensor, k: int) -> torch.Tensor:
    if k == 0:
        return x
    return torch.roll(x, shifts=int(k), dims=-1)

def _sample_permutation(cfg: Dict[str, Any] | None, C: int, g: Optional[torch.Generator], device: torch.device) -> Optional[torch.Tensor]:
    if not cfg or not cfg.get("enable", False):
        return None
    mode = cfg.get("mode", "random")
    if mode == "random":
        return torch.randperm(C, generator=g, device=device)
    elif mode == "fixed":
        path = cfg.get("perm_file", None)
        if path:
            try:
                arr = torch.load(path, map_location=device)
                perm = torch.as_tensor(arr, dtype=torch.long, device=device)
            except Exception:
                import numpy as np
                perm = torch.as_tensor(np.load(path), dtype=torch.long, device=device)
            assert perm.ndim == 1 and int(perm.numel()) == C, "perm_file must be a 1D permutation of size C"
            return perm
    return None

def make_handwriting_collate_with_channel_aug(base_collate, aug_cfg_split: Dict[str, Any] | None, seed: Optional[int] = None):
    """
    Wrap base handwriting_collate so we can apply channel rotation/permutation
    to batch['emg'] with shape (N, T, C). All other keys are untouched.
    """
    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))

    def _collate(samples):
        batch = base_collate(samples)
        x = batch.get("emg", None)
        if x is None:
            return batch
        C = int(x.shape[-1])
        device = x.device

        # rotation
        rot_cfg = (aug_cfg_split or {}).get("rotation", {})
        k = _sample_rotation_k(rot_cfg, C, gen)
        if rot_cfg.get("enable", False):
            x = _apply_rotation(x, k)

        # permutation
        perm_cfg = (aug_cfg_split or {}).get("permutation", {})
        if perm_cfg.get("enable", False):
            perm = _sample_permutation(perm_cfg, C, gen, device)
            if perm is not None:
                x = x.index_select(dim=-1, index=perm)

        batch["emg"] = x
        return batch

    return _collate

def custom_collate_fn(batch):
    """
    Custom collate function that handles pandas DataFrames and numpy arrays in batches.

    Parameters
    ----------
        batch: A list of dictionaries from __getitem__

    Returns
    -------
        A dictionary with batched tensors and non-tensor types
    """

    elem = batch[0]
    result = {}

    for key in elem:
        if isinstance(elem[key], pd.DataFrame):
            result[key] = [d[key] for d in batch]
        elif isinstance(elem[key], np.ndarray) and key == "timestamps":
            result[key] = [d[key] for d in batch]
        else:
            try:
                result[key] = default_collate([d[key] for d in batch])
            except TypeError:
                # Fallback for any other types that can't be collated
                result[key] = [d[key] for d in batch]

    return result


class WindowedEmgDataModule(pl.LightningDataModule):
    """A PyTorch LightningDataModule for constructing dataloaders to
    assemble batches of strided windows of contiguous sEMG

    Automatically takes care of applying random jitter to the windows
    used by the train dataloader, but not the validation and test dataloaders.

    The test dataloader is also enforced to return data over the full test set
    partitions, rather than over short windows within each partition. This is
    to emulate online application of these models, where inference is applied
    over the long timescale of an HCI task.

    Parameters
    ----------
    window_length : int
        Number of contiguous samples in each sample in the batch.
    stride : int | None
        Stride between consecutive windows from the same recording.
        Specify None to set this to window_length, in which case
        there will be no overlap between consecutive windows.
    batch_size : int
        The number of samples per batch.
    num_workers : int
        The number of subprocesses to use for data loading.
    data_split : DataSplit
        A dataclass containing a dictionary of datasets and
        corresponding partitions for the train, val, and test
        splits.
    transform : Transform
        A callable that takes a window/slice of `EmgRecording` in the
        form of a numpy structured array and a pandas DataFrame with
        prompt labels and times, and returns a `torch.Tensor` instance.
    data_location : str
        Path to where the dataset files are stored.
    emg_augmentation : Callable[[torch.Tensor], torch.Tensor], optional
        An optional function that takes an EMG tensor and returns
        an augmented EMG tensor. See augmentation.py.
    """

    def __init__(
        self,
        window_length: int,
        stride: int | None,
        batch_size: int,
        num_workers: int,
        data_split: DataSplit,
        transform: Transform,
        data_location: str,
        emg_augmentation: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.stride = stride
        self.data_split = data_split

        self.batch_size = batch_size
        self.transform = transform
        self.emg_augmentation = emg_augmentation
        self.num_workers = num_workers
        self.data_location = data_location

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = make_dataset(
                data_location=self.data_location,
                transform=self.transform,
                partition_dict=self.data_split.train,
                window_length=self.window_length,
                stride=self.stride,
                jitter=True,
                emg_augmentation=self.emg_augmentation,
                split_label="train",
            )
        if stage == "fit" or stage == "validate" or stage is None:
            self.val_dataset = make_dataset(
                data_location=self.data_location,
                transform=self.transform,
                partition_dict=self.data_split.val,
                window_length=self.window_length,
                stride=self.stride,
                jitter=False,
                emg_augmentation=None,
                split_label="val",
            )
        if stage == "test" or stage is None:
            self.test_dataset = make_dataset(
                data_location=self.data_location,
                transform=self.transform,
                partition_dict=self.data_split.test,
                # At test time, we feed in the entire partition in one
                # window to be more consistent with real-time deployment.
                window_length=None,
                stride=None,
                jitter=False,
                emg_augmentation=None,
                split_label="test",
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire partitions are
        # fed at once. Limit batch size to 1 to fit within GPU memory.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )


class HandwritingEmgDataModule(pl.LightningDataModule):
    """A PyTorch LightningDataModule for constructing dataloaders to
    assemble batches of handwriting EMG data.

    Automatically takes care of applying random jitter to the windows
    used by the train dataloader, but not to the validation and test dataloaders.

    The test dataloader is also enforced to return data over single partitions,
    rather than the longer windows (created during prompt concatenation) that
    are created when `concatenate_prompts=True`.

    Parameters
    ----------
    batch_size : int
        The number of samples per batch.
    padding : tuple[int, int]
        Zero-padding to apply to the beginning and end of the EMG data.
        This is dependent on the model architecture and should be used
        to ensure that the output of the model has the same length, no
        matter the input length. In other words it is used to account
        for the kernel size of the convolutional layers of the model.
    num_workers : int
        The number of subprocesses to use for data loading.
    data_split : DataSplit
        A dataclass containing a dictionary of datasets and
        corresponding partitions for the train, val, and test
        splits.
    transform : HandwritingTransform
        A callable that takes a window/slice of `EmgRecording` in the
        form of a numpy structured array and a prompt string, and
        returns a dictionary with "emg" and "prompts" keys.
    data_location : str
        Path to where the dataset files are stored.
    emg_augmentation : Callable[[torch.Tensor], torch.Tensor] | None
        An optional function that takes an EMG tensor and returns
        an augmented EMG tensor.
    concatenate_prompts : bool
        Whether to perform concatenation of multiple prompt samples
        up to `min_duration_s` seconds. This is useful to improve
        model performance and training stability.
    min_duration_s : float
        Minimum duration of the EMG recording in seconds when using
        `concatenate_prompts=True`. Ignored otherwise.
    """

    def __init__(
        self,
        batch_size: int,
        padding: tuple[int, int],
        num_workers: int,
        data_split: DataSplit,
        transform: HandwritingTransform,
        data_location: str,
        emg_augmentation: Callable[[torch.Tensor], torch.Tensor] | None = None,
        concatenate_prompts: bool = False,
        min_duration_s: float = 0.0,
        channel_aug: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.collate_fn = handwriting_collate
        self.channel_aug = channel_aug or {}
        self.seed = seed

        self.batch_size = batch_size
        self.padding = padding
        self.num_workers = num_workers
        self.data_split = data_split
        self.transform = transform
        self.emg_augmentation = emg_augmentation
        self.data_location = data_location
        self.concatenate_prompts = concatenate_prompts
        self.min_duration_s = min_duration_s

    def _collate_for_split(self, split: str):
        aug_cfg = (self.channel_aug or {}).get(split, {})
        return make_handwriting_collate_with_channel_aug(self.collate_fn, aug_cfg, seed=self.seed)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = make_handwriting_dataset(
                data_location=self.data_location,
                transform=self.transform,
                padding=self.padding,
                dataset_names=list(self.data_split.train.keys()),
                emg_augmentation=self.emg_augmentation,
                concatenate_prompts=self.concatenate_prompts,
                min_duration_s=self.min_duration_s,
                split_label="train",
            )
        if stage == "fit" or stage == "validate" or stage is None:
            self.val_dataset = make_handwriting_dataset(
                data_location=self.data_location,
                transform=self.transform,
                padding=self.padding,
                dataset_names=list(self.data_split.val.keys()),
                emg_augmentation=None,
                concatenate_prompts=False,
                min_duration_s=0.0,
                split_label="val",
            )
        if stage == "test" or stage is None:
            self.test_dataset = make_handwriting_dataset(
                data_location=self.data_location,
                transform=self.transform,
                padding=self.padding,
                dataset_names=list(self.data_split.test.keys()),
                emg_augmentation=None,
                concatenate_prompts=False,
                min_duration_s=0.0,
                split_label="test",
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=self._collate_for_split("train"),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self._collate_for_split("val"),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=self._collate_for_split("test"),
        )
