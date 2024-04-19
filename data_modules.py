# Imports

# System
import os
# Utils
import pickle
import numpy as np
from typing import List
from enum import Enum
# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
# Torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from pytorch_lightning import LightningDataModule
# Customs
from datasets import ImageFolderDataset

# Enum for different sampling strategies
class Sampling(Enum):
    NUMPY = 1
    SKLEARN = 2
    NONE = 3

# Utility class for managing indices
class IndexManager:
    @staticmethod
    def save_indices(indices: List[int], indices_path: str):
        """
        Save indices to a file.

        Args:
            indices (tuple): Tuple containing train and test indices.
            indices_path (str): Path to the file where indices will be saved.
        """
        with open(indices_path, "wb") as file:
            pickle.dump(indices, file)

    @staticmethod
    def load_indices(indices_path: str):
        """
        Load indices from a file.

        Args:
            indices_path (str): Path to the file containing saved indices.

        Returns:
            tuple: Tuple containing train and test indices.
        """
        with open(indices_path, "rb") as file:
            return pickle.load(file)

# Utility class for splitting data into train and test sets
class DataSplitter:
    @staticmethod
    def split_data(folder_dataset: ImageFolderDataset, indices_path: str, test_size: float, use_index: bool):
        """
        Split data into train and test indices.

        Args:
            folder_dataset (Dataset): Dataset instance.
            indices_path (str): Path to the file where indices will be saved.
            test_size (float): Fraction of the data to reserve as test set.
            use_index (bool): Flag indicating whether to use existing indices.

        Returns:
            tuple: Tuple containing train and test indices.
        """
        if use_index:
            return IndexManager.load_indices(indices_path)
        else:
            indices = train_test_split(
                range(len(folder_dataset)),
                test_size=test_size,
                stratify=folder_dataset.labels,
            )
            IndexManager.save_indices(indices, indices_path)
            return indices

# Utility class for creating data loaders
class DataLoaderCreator:
    @staticmethod
    def create_dataloader(dataset: Dataset, sampler=None, shuffle: bool=False, num_workers: int=1):
        """
        Create a DataLoader for a dataset.

        Args:
            dataset (Dataset): Dataset instance.
            sampler (optional): Sampler used for sampling data. Default is None.
            shuffle (bool, optional): Flag indicating whether to shuffle the data. Default is False.
            num_workers (int, optional): Number of subprocesses to use for data loading. Default is 1.

        Returns:
            DataLoader: DataLoader instance.
        """
        return DataLoader(
            dataset,
            batch_size=dataset.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

class SamplerFactory:
    @staticmethod
    def create_sampler(sampling: Sampling, train_dataset: Dataset, train_labels):
        """
        Create a sampler based on the specified sampling strategy.

        Args:
            sampling (Sampling): Enum value indicating the sampling strategy.
            train_dataset (Dataset): Training dataset.
            train_labels (optional): Train labels.

        Returns:
            Sampler or None: Sampler instance based on the specified strategy, or None if no sampler is needed.
        """
        if sampling == Sampling.NONE:
            return None
    
        elif sampling == Sampling.NUMPY:
            class_counts = np.array(
                [np.sum(train_labels == c) for c in np.unique(train_labels)]
            )
            class_weights = 1 / class_counts
                
            return WeightedRandomSampler(class_weights, len(train_dataset))
        else:
            class_weights = class_weight.compute_class_weight(
                class_weight="balanced", classes=np.unique(train_labels), y=train_labels
            )
            return WeightedRandomSampler(class_weights, len(train_dataset))


class ImagesDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: str,
        root_dir: str,
        batch_size: int,
        test_size:float =0.5,
        use_index:bool =True,
        indices_dir:str =None,
        sampling:Sampling=Sampling.NONE,
        train_transform=None,
        test_transform=None
    ):
        """
        Initialize the ImageDataModule.

        Args:
            dataset (str): Name of the dataset.
            root_dir (str): Root directory of the dataset.
            batch_size (int): Batch size for data loaders.
            test_size (float, optional): Fraction of data to use as test set. Default is 0.5.
            use_index (bool, optional): Whether to use existing indices. Default is True.
            indices_dir (str, optional): Directory to save indices. Default is None.
            sampling (Sampling, optional): Sampling strategy. Default is Sampling.NONE.
            train_transform (optional): Transformations to apply to training data. Default is None.
            test_transform (optional): Transformations to apply to test data. Default is None.
        """
        super().__init__()
        self.save_hyperparameters()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.test_size = test_size
        self.use_index = use_index
        self.sampling = sampling
        
        # Initialize training and test folders
        self.train_folder = ImageFolderDataset(
            root=root_dir, transform=train_transform
        )
        self.test_folder = ImageFolderDataset(
            root=root_dir, transform=test_transform
        )
        
        self.class_counts = self.train_folder.class_counts
        self.indices_path = os.path.join(indices_dir, str(dataset) + ".pkl")
        
    def prepare_data(self):
        """
        Prepare data for training and testing.
        """
        # Split train and test indices
        self.train_indices, self.test_indices = DataSplitter.split_data(self.train_folder, self.indices_path, self.test_size, self.use_index)
        # Split the datasets
        self.train_dataset = Subset(self.train_folder, self.train_indices)
        self.test_dataset = Subset(self.test_folder, self.test_indices)
        train_labels = np.array(self.train_folder.targets)[self.train_indices]
        # Create a sampler (if needed)
        self.train_sampler = SamplerFactory.create_sampler(self.sampling, self.train_dataset, train_labels)
        
    def create_data_loaders(self):
        """
        Create data loaders for training and testing.
        """
        # Shuffle flag
        shuffle = True if self.sampling == Sampling.NONE else False
        # Create data loaders
        self.train_loader = DataLoaderCreator.create_dataloader(self.train_dataset, self.train_sampler, shuffle=shuffle, num_workers=8)
        self.test_loader = DataLoaderCreator.create_dataloader(self.test_dataset, num_workers=8)
        
    def train_dataloader(self):
        """
        Get the training data loader.

        Returns:
            DataLoader: Training data loader.
        """
        return self.train_loader

    def val_dataloader(self):
        """
        Get the validation data loader (same as test data loader).

        Returns:
            DataLoader: Validation data loader.
        """
        return self.test_loader

    def test_dataloader(self):
        """
        Get the test data loader.

        Returns:
            DataLoader: Test data loader.
        """
        return self.test_loader

# CR Leaves specific data module
class CRLeavesDataModule(ImagesDataModule):
    def __init__(
        self,
        root_dir: str,
        batch_size: int,
        test_size:float =0.5,
        use_index:bool =True,
        indices_dir:str =None,
        sampling:Sampling=Sampling.NONE,
        train_transform=None,
        test_transform=None
    ):
        """
        Initialize a CRLeaves dataset data module.

        Args:
            root_dir (str): Root directory of the dataset.
            batch_size (int): Batch size for data loaders.
            test_size (float, optional): Fraction of data to use as test set. Default is 0.5.
            use_index (bool, optional): Whether to use existing indices. Default is True.
            indices_dir (str, optional): Directory to save indices. Default is None.
            sampling (Sampling, optional): Sampling strategy. Default is Sampling.NONE.
            train_transform (optional): Transformations to apply to training data. Default is None.
            test_transform (optional): Transformations to apply to test data. Default is None.
        """
        super().__init__(
            dataset="CRLeaves",
            root_dir=root_dir,
            batch_size=batch_size,
            test_size=test_size,
            use_index=use_index,
            indices_dir=indices_dir,
            sampling=sampling,
            train_transform=train_transform,
            test_transform=test_transform
        )
        