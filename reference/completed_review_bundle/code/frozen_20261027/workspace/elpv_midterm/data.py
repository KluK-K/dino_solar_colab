"""Integrity checks, an immutable split, and development-only feature extraction."""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from skimage.feature import hog
from skimage.transform import resize

ROOT = Path(__file__).resolve().parent


def config():
    return json.loads((ROOT / "config.json").read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixels(path):
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (300, 300):
            raise ValueError(f"Unexpected image format: {path}")
        return np.asarray(image).copy()


def prepare():
    cfg = config()
    source = ROOT / "data/source"
    manifest_path = ROOT / "data/split_manifest.json"
    signature = {
        "seed": cfg["seed"], "test_fraction": cfg["test_fraction"],
        "target": cfg["target"], "labels_sha256": sha256(source / "labels.csv"),
        "source_commit": json.loads((ROOT / "data/provenance.json").read_text())["commit"],
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["signature"] != signature:
            raise RuntimeError("Split is immutable; configuration or labels changed.")
        print("Existing split retained; test remains sealed.", flush=True)
        return old
    table = pd.read_csv(source / "labels.csv", sep=r"\s+", header=None,
                        names=["path", "expert_label", "cell_type"])
    if len(table) != 2624 or table.isna().any().any() or table.path.duplicated().any():
        raise ValueError("Unexpected labels or duplicate paths")
    if set(table.cell_type) != {"mono", "poly"}:
        raise ValueError("Unexpected cell types")
    if not np.allclose(np.sort(table.expert_label.unique()), [0, 1/3, 2/3, 1]):
        raise ValueError("Unexpected expert labels")
    table["target"] = (table.expert_label > 0).astype(int)
    table["confidence_weight"] = np.where(table.expert_label.isin([0, 1]), 1, table.expert_label)
    table["stratum"] = table.cell_type + "_" + (table.expert_label * 3).round().astype(int).astype(str)
    # This reads pixels only for integrity/duplicate checks, before test allocation.
    hashes = []
    for name in table.path:
        hashes.append(hashlib.sha256(pixels(source / name).tobytes()).hexdigest())
    table["pixel_sha256"] = hashes
    duplicate_count = int(table.pixel_sha256.duplicated().sum())
    if duplicate_count:
        grouped = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=cfg["seed"])
        dev, test = next(grouped.split(table, table.stratum, table.pixel_sha256))
    else:
        dev, test = train_test_split(np.arange(len(table)), test_size=cfg["test_fraction"],
                                    stratify=table.stratum, random_state=cfg["seed"])
    if set(table.iloc[dev].pixel_sha256) & set(table.iloc[test].pixel_sha256):
        raise AssertionError("Identical pixels cross the split")
    manifest = {
        "signature": signature,
        "development_ids": sorted(table.iloc[dev].path.tolist()),
        "test_ids": sorted(table.iloc[test].path.tolist()),
        "test_state": "sealed",
        "duplicate_extra_images": duplicate_count,
        "split_method": "stratified_by_expert_label_and_type" if not duplicate_count else "stratified_exact_duplicate_groups",
        "limitation": "44 module identities unavailable; pixel groups are NOT module groups.",
    }
    table.to_csv(ROOT / "data/metadata.csv", index=False)
    save_json(manifest_path, manifest)
    audit = {
        "images": len(table), "image_size": [300, 300], "image_mode": "L",
        "expert_label_counts": {str(k): int(v) for k, v in table.expert_label.value_counts().items()},
        "target_counts": {str(k): int(v) for k, v in table.target.value_counts().items()},
        "cell_type_counts": {str(k): int(v) for k, v in table.cell_type.value_counts().items()},
        "duplicate_extra_images": duplicate_count,
        "development_size": len(dev), "test_size": len(test),
        "test_state": "sealed", "module_ids_available": False,
    }
    save_json(ROOT / "data/audit.json", audit)
    print(f"Integrity passed: {len(table)} images; development={len(dev)}, sealed test={len(test)}; duplicates={duplicate_count}", flush=True)
    return manifest


def load_partition(partition="development", *, authorized_test=False):
    if partition not in {"development", "test"}:
        raise ValueError("Unknown partition")
    if partition == "test" and not (authorized_test and config()["checkpoint_passed"]):
        raise PermissionError("Test is sealed until after checkpoint. Use final_evaluate.py later.")
    manifest = json.loads((ROOT / "data/split_manifest.json").read_text())
    table = pd.read_csv(ROOT / "data/metadata.csv")
    ids = manifest[f"{partition}_ids"]
    return table.set_index("path").loc[ids].reset_index()


def stats_features(image):
    gray = image.astype(np.float64) / 255.0
    dy, dx = np.gradient(gray)
    gradient = np.hypot(dx, dy)
    values = [gray.mean(), gray.std(), *np.quantile(gray, [0.1, 0.5, 0.9]),
              (gray < .15).mean(), (gray < .30).mean(),
              gradient.mean(), gradient.std(), np.quantile(gradient, .9)]
    for row in np.array_split(gray, 4, axis=0):
        for block in np.array_split(row, 4, axis=1):
            values.extend([block.mean(), block.std()])
    return np.asarray(values)


def feature_signature():
    cfg = config()
    settings = {k: v for k, v in cfg.items() if k.startswith("hog_")}
    return {"settings": settings, "feature_code_sha256": sha256(__file__)}


def extract(partition="development", *, authorized_test=False, force=False):
    table = load_partition(partition, authorized_test=authorized_test)
    cfg = config()
    cache = ROOT / f"cache/{partition}_features.npz"
    meta = ROOT / f"cache/{partition}_features.json"
    signature = feature_signature()
    if not force and cache.exists() and meta.exists():
        metadata = json.loads(meta.read_text())
        if metadata["signature"] == signature and metadata["ids"] == table.path.tolist():
            with np.load(cache) as values:
                return table, values["stats"], values["hog"], metadata
    started = time.perf_counter()
    statistics, gradients = [], []
    for number, row in enumerate(table.itertuples(), 1):
        image = pixels(ROOT / "data/source" / row.path)
        if hashlib.sha256(image.tobytes()).hexdigest() != row.pixel_sha256:
            raise ValueError(f"Image changed since integrity audit: {row.path}")
        statistics.append(stats_features(image))
        small = resize(image, (cfg["hog_size"], cfg["hog_size"]), anti_aliasing=True)
        gradients.append(hog(small, orientations=cfg["hog_orientations"],
                             pixels_per_cell=(cfg["hog_pixels_per_cell"],) * 2,
                             cells_per_block=(cfg["hog_cells_per_block"],) * 2,
                             block_norm="L2-Hys", transform_sqrt=False))
        if number % 400 == 0:
            print(f"Features {partition}: {number}/{len(table)}", flush=True)
    a, b = np.asarray(statistics), np.asarray(gradients)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite features")
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, stats=a, hog=b)
    metadata = {
        "signature": signature, "ids": table.path.tolist(), "partition": partition,
        "stats_dimensions": a.shape[1], "hog_dimensions": b.shape[1],
        "extraction_seconds": time.perf_counter() - started,
        "learned_preprocessing": "None; StandardScaler is fitted inside model CV pipelines.",
    }
    save_json(meta, metadata)
    print(f"Features ready: {a.shape[1]} statistics + {b.shape[1]} HOG; {metadata['extraction_seconds']:.1f}s", flush=True)
    return table, a, b, metadata


def select_features(stats, gradients, representation="hog_stats"):
    if representation == "stats":
        return stats
    if representation == "hog_stats":
        return np.column_stack([stats, gradients])
    raise ValueError("Unknown representation")


if __name__ == "__main__":
    prepare()
    extract()
