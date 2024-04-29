# Imports

# System
import os

# Torch
from torch.utils.data import Dataset

# Utils
from collections import defaultdict
from typing import List, Tuple

# Images
from PIL import Image


# Utility class for scanning a folder contents
class FolderScanner:
    @staticmethod
    def count_files(root_dir: str) -> dict:
        """
        Count the number of files in each folder within the specified root directory.

        Args:
            root_dir (str): The root directory to scan.

        Returns:
            dict: A dictionary where keys are folder names and values are the number of files in each folder.
        """
        folder_counts = {}
        for folder in os.scandir(root_dir):
            if folder.is_dir():
                folder_counts[folder.name] = sum(
                    1 for _ in os.scandir(folder) if _.is_file()
                )

        # Sort folders based on file counts
        return sorted(folder_counts, key=folder_counts.get)


# Utility class for building images and labels lists
class ImageListBuilder:
    @staticmethod
    def build_list(
        root_dir: str, folders: List[str]
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        Build lists of image paths, corresponding labels, and class counts.

        Args:
            root_dir (str): The root directory containing image folders.
            folders (List[str]): List of folder names containing images.

        Returns:
            Tuple[List[str], List[int], List[int]]: A tuple containing lists of image paths, labels, and class counts.
        """
        images = []
        labels = []
        class_counts = defaultdict(int)
        classes = {}
        for i, folder in enumerate(folders):
            folder_path = os.path.join(root_dir, folder)
            classes[folder] = i

            for image_name in os.scandir(folder_path):
                if (
                    image_name.name.lower().endswith(("jpg", "jpeg", "png"))
                    and image_name.is_file()
                ):
                    images.append(os.path.join(folder_path, image_name.name))
                    labels.append(i)
                    class_counts[i] += 1

        return images, labels, list(class_counts.values()), classes


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        """
        Initialize the ImageFolderDataset.

        Args:
            root_dir (str): The root directory containing image folders.
            transform (optional): An optional transform to be applied to the images.
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        # Count files in each folder
        self.folders = FolderScanner.count_files(root_dir=self.root_dir)

        # Build lists of images, labels, and class counts
        self.images, self.labels, self.class_counts, self.classes = (
            ImageListBuilder.build_list(root_dir=self.root_dir, folders=self.folders)
        )

    def __len__(self):
        """
        Get the total number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get an image and its corresponding label at the specified index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: A tuple containing the image and its label.
        """
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
