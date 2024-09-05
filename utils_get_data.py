import requests
import os
from pathlib import Path
import tarfile
import glob
import json
from collections import Counter

from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_directory_size(directory_path: Path) -> int:
    """
    Calculates the total size of a directory in megabytes.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return round(total_size / (1024 * 1024), 2)


# download data from reuters21578.tar.gz
# first check if the file on the link exists, if it exists then download it
# also check if the file already exists in the destination location


def download_data(
    url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz",
    filename: str = "reuters21578.tar.gz",
    destination: Path = Path("data"),
) -> None:
    """
    Downloads the dataset from the given URL to the specified destination.
    Args:
        url (str): URL of the dataset file.
        filename (str): Name of the downloaded file.
        destination (Path): Destination directory where the downloaded file will be saved.
    Returns:
        None
    """
    # check if filename exists in destination
    if not destination.is_dir():
        destination.mkdir(parents=True, exist_ok=True)

        # check if url exists and download file if it does
        if requests.head(url).status_code == 200:
            print(f"Downloading...")
            response = requests.get(url, stream=True)
            # download file from url and save to destination
            with open(destination / filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        f.flush()
            print(f"File {filename} downloaded successfully.")
        else:
            print(f"Error: Unable to retrieve file from {url}")
    else:
        print(f"File {filename} already exists.")


def extract_file(
    filepath: Path = Path("data/reuters21578.tar.gz"),
    destination: Path = Path("data/reuters21578"),
) -> None:
    """
    Extracts the downloaded file to the specified destination.
    Args:
        filepath (Path): Path to the downloaded file.
        destination (Path): Destination directory where the extracted files will be saved.
    Returns:
        None
    """
    # check if the filepath (Path) exists and is a .tar.gz file
    if filepath.is_file() and filepath.suffixes == [".tar", ".gz"]:
        # check if destination path exists and is not empty.
        if not destination.is_dir() or not destination.glob("*"):
            print(f"Extracting {filepath.name}...")
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(destination)
            print(f"File {filepath.name} extracted successfully to {destination}.")
        else:
            print(f"Warning: Destination directory {destination} is not empty.")
            print(f"Size of {destination}: {get_directory_size(destination)} MB.")
    else:
        print(f"Error: {filepath.name} is not a valid .tar.gz file.")


def extract_data(
    pattern: str = "data/reuters21578/reut2-*.sgm",
    output_filepath: Path = Path("data/reuters21578.json"),
) -> pd.DataFrame:
    """
    Extracts titles, bodies, and topics from the SGML files in the specified pattern.
    Returns a dataframe with the extracted data.
    Args:
        pattern (str): The pattern to match SGML files.
        output_filepath (Path): The path where the extracted data will be saved as a JSON file.
    Returns:
        pd.DataFrame: A dataframe containing the extracted titles, bodies, and topics.
    """
    results = {
        "title": [],
        "body": [],
        "topics": [],
        "reuters_lewissplit": [],
    }

    if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
        print(f"Output file {output_filepath} already exists!")
        print(f"loading output file {output_filepath}...")
        with open(output_filepath, "r", encoding="utf-8") as file:
            results = json.load(file)
    else:
        print(f"Extracting Data to output file {output_filepath}...")
        for file_path in glob.glob(pattern):
            try:
                with open(file_path, "r", encoding="latin-1") as file:
                    sgml_content = file.read()

                # Parse the content with BeautifulSoup
                soup = BeautifulSoup(sgml_content, "html.parser")

                for reuters_tag in soup.find_all("reuters"):
                    # Extract title
                    title = reuters_tag.find("title")
                    if title:
                        results["title"].append(title.get_text(strip=True))
                    else:
                        results["title"].append(None)

                    # Extract body
                    body = reuters_tag.find("body")
                    if body:
                        results["body"].append(body.get_text(strip=True))
                    else:
                        results["body"].append(None)

                    # Extract topics
                    topics_tag = reuters_tag.find("topics")
                    topics = (
                        [
                            d_tag.get_text(strip=True)
                            for d_tag in topics_tag.find_all("d")
                        ]
                        if topics_tag
                        else []
                    )
                    results["topics"].append(topics)

                    # Extract Reuters attributes
                    lewissplit = reuters_tag.get("lewissplit")
                    results["reuters_lewissplit"].append(lewissplit)
                    # reuters_attrs = reuters_tag.attrs
                    # results["reuters"].append(reuters_attrs)

                with open(output_filepath, "w", encoding="utf-8") as out_file:
                    json.dump(results, out_file, ensure_ascii=False, indent=4)

            except UnicodeDecodeError as e:
                print(f"Error decoding {file_path}: {e}")
                return None

    return pd.DataFrame(results)


def create_train_test_split(df: pd.DataFrame) -> tuple:
    """
    Splits the dataset into train, validation, and test sets based on the 'lewissplit' attribute.
    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
    Returns:
        tuple: A tuple containing the train, validation, and test DataFrames.
    """

    df = df.rename(columns={"body": "text", "topics": "labels"})

    print(f"Documents with no labels: {len(df[df['labels'].apply(len) == 0])}")
    print(f"Documents with labels: {len(df[df['labels'].apply(len) != 0])}")
    print(f"Total Documents: {len(df)}")

    labels_df = df[df["labels"].apply(len) != 0]
    train_df = labels_df[labels_df["reuters_lewissplit"] == "TRAIN"]
    test_df = labels_df[labels_df["reuters_lewissplit"] == "TEST"]
    print(f"Train Documents: {len(train_df)}")
    print(f"Test Documents: {len(test_df)}")

    # Separate features and labels
    X = train_df["text"]
    y = train_df["labels"]

    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=1000, random_state=42
    )

    train_df = pd.DataFrame({"text": X_train, "labels": y_train})
    val_df = pd.DataFrame({"text": X_val, "labels": y_val})

    return train_df, val_df, test_df


def get_labels(df: pd.DataFrame) -> tuple:
    """
    Returns a tuple containing the unique labels, label counts, and label-to-index mapping.
    Args:
        df (pd.DataFrame): DataFrame containing the dataset.
    Returns:
        tuple: Unique labels, label counts, label-to-index mapping, and index-to-label mapping.
    """

    all_labels = [label for labels in df["labels"] for label in labels]

    unique_labels = pd.unique(all_labels)
    print(f"Total labels: {len(unique_labels)}")
    print(f"labels: {unique_labels}")

    label_counts = Counter(all_labels)
    label_counts_df = pd.DataFrame(
        label_counts.items(), columns=["label", "count"]
    ).sort_values("count", ascending=False)

    # Create a mapping from labels to numerical values
    label2idx = {
        label: idx for idx, label in enumerate(label_counts_df["label"].unique())
    }
    idx2label = {
        idx: label for idx, label in enumerate(label_counts_df["label"].unique())
    }

    # Map the 'Label' column to numerical values
    label_counts_df["label_num"] = label_counts_df["label"].map(label2idx)

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(
        label_counts_df["label_num"],
        label_counts_df["count"],
        tick_label=label_counts_df["label_num"],
    )
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.title("Count of Each Label")
    plt.xticks([])  # Set x-ticks to show labels
    plt.show()

    return unique_labels, label_counts, label2idx, idx2label
