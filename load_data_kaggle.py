# load_data_kaggle.py

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]

import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_kaggle_dataset(file_path=""):
    """
    Load the Breast Cancer Wisconsin dataset from Kaggle using kagglehub.

    Args:
        file_path (str): Optional specific file path in the Kaggle dataset.
                         Leave empty to load the default file.

    Returns:
        pandas.DataFrame: Loaded dataset
    """
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "uciml/breast-cancer-wisconsin-data",
        file_path
    )
    return df


if __name__ == "__main__":
    df = load_kaggle_dataset()
    print("First 5 records:\n", df.head())
