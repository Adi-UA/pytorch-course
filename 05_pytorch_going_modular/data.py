from pathlib import Path
from zipfile import ZipFile

import requests
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

DATA_DIR = Path("data")


def fetch_data(file_name: str, url: str):
    print("Downloading images...")
    r = requests.get(url)

    if r.status_code != 200:
        raise RuntimeError(f"Could not fetch {url} | Error {r.status_code}")

    file_path = DATA_DIR / file_name
    with open(file_path, "wb") as f:
        f.write(r.content)

    print("Images downloaded")


def unzip_data(file_name: str, extract_dir: str):
    print("Unzipping images...")

    file_path = DATA_DIR / file_name
    extract_dir = DATA_DIR / extract_dir

    with ZipFile(file_path, "r") as zip_file:
        zip_file.extractall(extract_dir)

    print("Images unzipped")


def create_data_loaders(
    train_dir, test_dir, train_transform, test_transform, batch_size
):
    train_dir, test_dir = DATA_DIR / train_dir, DATA_DIR / test_dir
    train_data = ImageFolder(train_dir, transform=train_transform)
    test_data = ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
