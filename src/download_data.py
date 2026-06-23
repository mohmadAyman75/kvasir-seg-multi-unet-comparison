from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests
import urllib3
from tqdm import tqdm


DATA_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"


def download_file(url: str, output_path: Path, verify_ssl: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True, timeout=30, verify=verify_ssl)
    except requests.exceptions.SSLError:
        if not verify_ssl:
            raise
        print("SSL certificate verification failed. Retrying without certificate verification...")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(url, stream=True, timeout=30, verify=False)

    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with output_path.open("wb") as file, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {output_path.name}",
    ) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
                progress.update(len(chunk))


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)


def prepare_kvasir(output_dir: Path, keep_zip: bool = False) -> Path:
    dataset_dir = output_dir / "Kvasir-SEG"
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"

    if images_dir.exists() and masks_dir.exists():
        print(f"Dataset already exists: {dataset_dir}")
        return dataset_dir

    zip_path = output_dir / "Kvasir-SEG.zip"
    if not zip_path.exists():
        download_file(DATA_URL, zip_path)

    print(f"Extracting {zip_path} ...")
    extract_zip(zip_path, output_dir)

    if not keep_zip and zip_path.exists():
        zip_path.unlink()

    if not images_dir.exists() or not masks_dir.exists():
        raise RuntimeError(
            "Could not find images/ and masks/ after extraction. "
            f"Check the extracted files in: {output_dir}"
        )

    print(f"Ready: {dataset_dir}")
    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Kvasir-SEG dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--keep-zip", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_kvasir(args.output_dir, keep_zip=args.keep_zip)
