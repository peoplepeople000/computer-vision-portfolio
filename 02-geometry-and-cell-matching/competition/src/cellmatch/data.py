"""Dataset contracts and submission validation."""
import csv
import json
from pathlib import Path
import numpy as np
import tifffile
from .rle import Mask

COLUMNS = ["sample_id", "invivo_instances", "exvivo_instances", "match_pairs"]
csv.field_size_limit(100_000_000)


def split_id(sample_id):
    subject, sep, region = sample_id.partition("__")
    if not sep or not subject or not region or any(c in sample_id for c in "/\\") or subject in (".", "..") or region in (".", ".."):
        raise ValueError(f"Invalid sample ID: {sample_id!r}")
    return subject, region


def read_rows(path):
    rows = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Expected columns {COLUMNS}, got {reader.fieldnames}")
        for raw in reader:
            if None in raw or any(v is None for v in raw.values()):
                raise ValueError("Malformed CSV row")
            sid = raw["sample_id"]
            split_id(sid)
            if sid in rows:
                raise ValueError(f"Duplicate sample: {sid}")
            row = {"sample_id": sid}
            for key in COLUMNS[1:]:
                row[key] = json.loads(raw[key])
            validate_structure(row)
            rows[sid] = row
    if not rows:
        raise ValueError("CSV contains no samples")
    return rows


def validate_structure(row):
    for field in COLUMNS[1:3]:
        value = row[field]
        if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            raise ValueError(f"{field} must map string IDs to RLE strings")
    if not isinstance(row["match_pairs"], list):
        raise ValueError("match_pairs must be a JSON list")
    used_iv, used_ex = set(), set()
    for pair in row["match_pairs"]:
        if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(v, str) for v in pair):
            raise ValueError("Each pair must contain two string IDs")
        iv, ex = pair
        if iv not in row["invivo_instances"] or ex not in row["exvivo_instances"]:
            raise ValueError("Pair references an undefined cell")
        if iv in used_iv or ex in used_ex:
            raise ValueError("A cell appears in multiple pairs")
        used_iv.add(iv)
        used_ex.add(ex)


def read_image(path):
    image = tifffile.imread(path)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError(f"Expected a finite 2D grayscale TIFF: {path}, shape={image.shape}")
    return image


def image_path(data_dir, split, sid, modality):
    subject, region = split_id(sid)
    return Path(data_dir) / split / subject / region / f"{modality}.tif"


def image_shape(path):
    with tifffile.TiffFile(path) as tif:
        shape = tuple(tif.series[0].shape)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"Expected a 2D image: {path}, shape={shape}")
    return shape


def parse_instances(row, modality, shape):
    return {key: Mask.parse(value, shape) for key, value in row[f"{modality}_instances"].items()}


def write_rows(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            validate_structure(row)
            writer.writerow({key: row[key] if key == "sample_id" else json.dumps(row[key], separators=(",", ":")) for key in COLUMNS})
    temp.replace(path)


def validate_submission(data_dir, prediction_path):
    expected = read_rows(Path(data_dir) / "sample_submission.csv")
    actual = read_rows(prediction_path)
    if len(expected) != 29 or set(actual) != set(expected):
        raise ValueError("Submission must contain exactly the 29 official sample IDs")
    counts = {"regions": len(actual), "invivo": 0, "exvivo": 0, "pairs": 0}
    for sid, row in actual.items():
        for modality in ("invivo", "exvivo"):
            masks = parse_instances(row, modality, image_shape(image_path(data_dir, "hidden_test", sid, modality)))
            counts[modality] += len(masks)
        counts["pairs"] += len(row["match_pairs"])
    return counts
