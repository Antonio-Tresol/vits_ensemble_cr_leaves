import os
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image

def get_labels_in_dir(directory: str) -> list[str]:
    """
    Get the list of unique labels in the given directory.

    Args:
        directory (str): The directory path.

    Returns:
        list: A list of unique labels found in the directory.
    """
    unique_labels = [label for label in os.listdir(directory)]
    return unique_labels

def species_sample_count(directory: str) -> pd.DataFrame:
    """
    Get the count of samples per species.

    Args:
        directory (str): The directory path.

    Returns:
        list: The count of samples per species.
    """
    species_counter = {}
    for subdir in os.listdir(directory):
        # Set a new species counter
        species_counter[subdir] = sum([1 for filename in os.listdir(os.path.join(directory, subdir)) if (filename.endswith('.jpg') or filename.endswith('.JPG'))])
    species_counter = pd.DataFrame(data=species_counter.values(), columns=['Frecuency'], index=species_counter.keys())
    return species_counter

