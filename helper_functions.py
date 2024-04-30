import os


def count_classes(directory: str) -> int:
    """
    Get the list of unique labels in the given directory.

    Args:
        directory (str): The directory path.

    Returns:
        list: A list of unique labels found in the directory.
    """
    unique_labels = sum([1 for label in os.listdir(directory)])
    return unique_labels
