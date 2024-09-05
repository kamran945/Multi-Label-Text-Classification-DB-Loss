import requests
import os
from pathlib import Path
import tarfile
import glob
import json

from bs4 import BeautifulSoup


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
) -> dict:
    """
    Extracts titles, bodies, and topics from the SGML files in the specified pattern.
    Returns a dictionary with the extracted data.
    Args:
        pattern (str): The pattern to match SGML files.
        output_filepath (Path): The path where the extracted data will be saved as a JSON file.
    Returns:
        dict: A dictionary containing the extracted titles, bodies, and topics.
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

    return results
