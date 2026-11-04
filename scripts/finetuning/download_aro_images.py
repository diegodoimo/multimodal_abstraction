#!/usr/bin/env python3
"""Download the ARO annotations and the images of the ARO order subtasks (COCO val2014 + Flickr30k).

The ARO dataset code (dataset_zoo/) is vendored in third_party/vision-language-models-are-bows/; the
four JSON annotation files are downloaded from the sources that code uses (Google Drive for Visual
Genome, the BLIP release for the COCO / Flickr30k Karpathy test splits). The Visual Genome images are
fetched by the vendored code itself on the first run (download=True). Files already present are
skipped. Called by scripts/finetuning/download_aro.sh.

Usage:
    python scripts/finetuning/download_aro_images.py --aro-data-root DIR                          # everything
    python scripts/finetuning/download_aro_images.py --aro-data-root DIR --dataset annotations    # JSON only
    python scripts/finetuning/download_aro_images.py --aro-data-root DIR --dataset coco           # + COCO images
    python scripts/finetuning/download_aro_images.py --aro-data-root DIR --dataset flickr30k      # + Flickr30k images
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

# the sources used by third_party/vision-language-models-are-bows/dataset_zoo/aro_datasets.py
ANNOTATIONS = {
    "coco_karpathy_test.json": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json",
    "flickr30k_test.json": "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json",
    "visual_genome_relation.json": "gdrive:1kX2iCHEv0CADL8dSO1nMdW-V0NqIAiP3",
    "visual_genome_attribution.json": "gdrive:13tWvOrNOLHxl3Rm9cR3geAdHx2qR3-Tw",
}

COCO_IMAGE_URL = "http://images.cocodataset.org/val2014"
COCO_ZIP_URL = "http://images.cocodataset.org/zips/val2014.zip"

FLICKR_HF_REPO = "nlphuji/flickr30k"
FLICKR_HF_FILE = "flickr30k-images.zip"
FLICKR_MIN_IMAGES = 30000
COCO_DOWNLOAD_WORKERS = 16


def _count_jpgs(directory: Path) -> int:
    """Count .jpg files in a directory (non-recursive)."""
    if not directory.is_dir():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix == ".jpg")


def _download_one_image(args: tuple[str, Path]) -> bool:
    """Download a single image. Returns True on success."""
    url, dest = args
    if dest.exists():
        return True
    try:
        urlretrieve(url, dest)
        return True
    except Exception:
        return False


def download_annotations(aro_root: Path) -> None:
    """Download the four ARO annotation files that are not in aro_root yet."""
    for fname, source in ANNOTATIONS.items():
        dst = aro_root / fname
        if dst.exists():
            continue
        tmp = dst.with_suffix(".part")
        if source.startswith("gdrive:"):
            import gdown

            gdown.download(id=source.removeprefix("gdrive:"), output=str(tmp), quiet=True)
        else:
            urlretrieve(source, tmp)
        json.loads(tmp.read_text())  # fail on an HTML error page instead of a JSON file
        tmp.rename(dst)
        print(f"  Downloaded annotation: {fname}")


def download_coco_val2014(aro_root: Path, workers: int = COCO_DOWNLOAD_WORKERS) -> None:
    """Download the COCO val2014 images of the ARO annotations (about 5,000 images, 1 GB) in
    parallel; if any download fails, download the full val2014.zip (6 GB) instead."""
    target = aro_root / "val2014"

    ann_path = aro_root / "coco_karpathy_test.json"
    if not ann_path.exists():
        sys.exit(f"Annotation not found: {ann_path}")

    ann = json.loads(ann_path.read_text())
    needed_images = sorted({item["image"] for item in ann})  # e.g. val2014/COCO_...jpg
    print(f"COCO val2014: {len(needed_images)} images needed by annotations")

    target.mkdir(parents=True, exist_ok=True)
    present = {f"val2014/{f.name}" for f in target.iterdir() if f.suffix == ".jpg"}
    to_download = [img for img in needed_images if img not in present]

    if not to_download:
        print(f"COCO val2014: all {len(needed_images)} images already present")
        return

    print(f"  {len(present)} already present, {len(to_download)} to download")

    tasks = [
        (f"{COCO_IMAGE_URL}/{img.split('/')[-1]}", target / img.split("/")[-1])
        for img in to_download
    ]

    failed = 0
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one_image, t): t for t in tasks}
        for future in as_completed(futures):
            completed += 1
            if not future.result():
                failed += 1
            if completed % 500 == 0 or completed == len(tasks):
                print(f"  [{completed}/{len(tasks)}] downloaded ({failed} failed)")

    if failed > 0:
        print(f"{failed} images failed to download; downloading the full val2014.zip instead")
        _download_coco_zip_fallback(aro_root)
        return

    n_final = _count_jpgs(target)
    print(f"COCO val2014: {n_final} images")


def _download_coco_zip_fallback(aro_root: Path) -> None:
    """Download and extract the full val2014.zip."""
    zip_path = aro_root / "val2014.zip"
    target = aro_root / "val2014"

    print(f"Downloading {COCO_ZIP_URL} ...")
    subprocess.run(
        ["wget", "-c", "--progress=bar:force", COCO_ZIP_URL, "-O", str(zip_path)],
        check=True,
    )
    print("Extracting val2014.zip ...")
    subprocess.run(
        ["unzip", "-o", "-q", str(zip_path), "-d", str(aro_root)],
        check=True,
    )
    zip_path.unlink(missing_ok=True)

    n_images = _count_jpgs(target)
    print(f"COCO val2014: {n_images} images (from zip)")


def download_flickr30k(aro_root: Path) -> None:
    """Download the Flickr30k images from the Hugging Face Hub (4.3 GB zip, about 31,000 images)."""
    target = aro_root / "flickr30k-images"
    n_existing = _count_jpgs(target)
    if n_existing > FLICKR_MIN_IMAGES:
        print(f"Flickr30k: already present ({n_existing} images)")
        return

    print(f"Downloading Flickr30k from the Hugging Face Hub ({FLICKR_HF_REPO}) ...")

    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(
        FLICKR_HF_REPO, FLICKR_HF_FILE, repo_type="dataset"
    )

    print("Extracting flickr30k-images.zip ...")
    subprocess.run(
        ["unzip", "-o", "-q", zip_path, "-d", str(aro_root)],
        check=True,
        env={**os.environ, "UNZIP_DISABLE_ZIPBOMB_DETECTION": "TRUE"},
    )

    ann_path = aro_root / "flickr30k_test.json"
    if ann_path.exists():
        ann = json.loads(ann_path.read_text())
        needed = {item["image"].replace("flickr30k-images/", "") for item in ann}
        present_set = {f.name for f in target.iterdir() if f.suffix == ".jpg"}
        missing = needed - present_set
        if missing:
            print(f"Warning: {len(missing)} images of the annotations are missing from flickr30k-images/")

    n_images = _count_jpgs(target)
    if n_images < FLICKR_MIN_IMAGES:
        sys.exit(f"Expected more than {FLICKR_MIN_IMAGES} Flickr30k images, got {n_images}")

    print(f"Flickr30k: {n_images} images")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the ARO annotations and the images of the ARO order subtasks."
    )
    parser.add_argument(
        "--aro-data-root",
        type=Path,
        required=True,
        help="ARO data directory (ARO_DATA_ROOT of slurm/compositionality.sbatch)",
    )
    parser.add_argument(
        "--dataset",
        choices=["annotations", "coco", "flickr30k", "all"],
        default="all",
        help="What to download besides the annotations (default: all); 'annotations' downloads only the JSON files",
    )
    args = parser.parse_args()

    args.aro_data_root.mkdir(parents=True, exist_ok=True)

    download_annotations(args.aro_data_root)

    if args.dataset in ("coco", "all"):
        download_coco_val2014(args.aro_data_root)

    if args.dataset in ("flickr30k", "all"):
        download_flickr30k(args.aro_data_root)

    print(f"ARO data ready in {args.aro_data_root}")


if __name__ == "__main__":
    main()
