"""Discovery helpers for inference models and raw-image corpora."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .paths import input_root as default_input_root
from .paths import models_root as default_models_root

_TRUE_VALUES = {"1", "true", "t", "yes", "y"}


@dataclass(frozen=True)
class ModelRunArtifact:
    """One selectable trained run artifact."""

    label: str
    model_name: str
    run_id: str
    model_dir: Path
    run_dir: Path
    config_path: Path
    run_manifest_path: Path | None
    checkpoint_path: Path


@dataclass(frozen=True)
class RawCorpus:
    """One selectable raw-image corpus that matches the input-images contract."""

    name: str
    root: Path
    images_dir: Path
    run_json_path: Path
    samples_csv_path: Path


def discover_model_runs(root: Path | None = None) -> list[ModelRunArtifact]:
    """Discover runnable model artifacts under `./models`."""
    models_root = (root or default_models_root()).resolve()
    if not models_root.exists():
        raise FileNotFoundError(f"Models root does not exist: {models_root}")

    artifacts: list[ModelRunArtifact] = []
    for config_path in sorted(models_root.glob("*/runs/run_*/config.json")):
        run_dir = config_path.parent.resolve()
        model_dir = run_dir.parent.parent.resolve()
        checkpoint_path = next(
            (
                candidate
                for candidate in (run_dir / "best.pt", run_dir / "best_model.pt")
                if candidate.exists()
            ),
            None,
        )
        if checkpoint_path is None:
            continue
        run_manifest_path = run_dir / "run_manifest.json"
        artifacts.append(
            ModelRunArtifact(
                label=f"{model_dir.name} / {run_dir.name}",
                model_name=model_dir.name,
                run_id=run_dir.name,
                model_dir=model_dir,
                run_dir=run_dir,
                config_path=config_path.resolve(),
                run_manifest_path=(run_manifest_path.resolve() if run_manifest_path.exists() else None),
                checkpoint_path=checkpoint_path.resolve(),
            )
        )
    return artifacts


def discover_raw_corpora(root: Path | None = None) -> list[RawCorpus]:
    """Discover only raw-image corpora under `./input`."""
    input_root = (root or default_input_root()).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    corpora: list[RawCorpus] = []
    for corpus_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        manifests_dir = corpus_dir / "manifests"
        images_dir = corpus_dir / "images"
        run_json_path = manifests_dir / "run.json"
        samples_csv_path = manifests_dir / "samples.csv"
        if not images_dir.exists():
            continue
        if not run_json_path.is_file() or not samples_csv_path.is_file():
            continue
        corpora.append(
            RawCorpus(
                name=corpus_dir.name,
                root=corpus_dir.resolve(),
                images_dir=images_dir.resolve(),
                run_json_path=run_json_path.resolve(),
                samples_csv_path=samples_csv_path.resolve(),
            )
        )
    return corpora


def _capture_success_mask(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .isin(_TRUE_VALUES)
    )


def load_corpus_samples(corpus: RawCorpus) -> pd.DataFrame:
    """Load selectable sample rows from one raw-image corpus manifest."""
    samples_df = pd.read_csv(corpus.samples_csv_path, low_memory=False)
    if samples_df.empty:
        return samples_df
    if "image_filename" not in samples_df.columns:
        raise ValueError(f"Corpus samples.csv is missing image_filename: {corpus.samples_csv_path}")
    if "capture_success" not in samples_df.columns:
        raise ValueError(f"Corpus samples.csv is missing capture_success: {corpus.samples_csv_path}")

    image_names = samples_df["image_filename"].astype("string").fillna("").str.strip()
    valid_mask = _capture_success_mask(samples_df["capture_success"]) & image_names.ne("")

    filtered = samples_df.loc[valid_mask].copy()
    filtered["__row_index__"] = filtered.index.astype(int)
    filtered["__image_name__"] = filtered["image_filename"].astype(str).str.strip()
    filtered["__image_path__"] = filtered["__image_name__"].map(
        lambda name: str((corpus.images_dir / name).resolve())
    )
    filtered = filtered.loc[filtered["__image_path__"].map(lambda value: Path(value).is_file())].copy()
    return filtered.reset_index(drop=True)


def list_corpus_image_names(corpus: RawCorpus) -> list[str]:
    """Return selectable image filenames for one corpus."""
    samples_df = load_corpus_samples(corpus)
    if samples_df.empty:
        return []
    duplicates = samples_df["__image_name__"].duplicated(keep=False)
    if duplicates.any():
        examples = samples_df.loc[duplicates, "__image_name__"].tolist()[:5]
        raise ValueError(
            "Corpus image selection requires unique image_filename values. "
            f"Found duplicates in {corpus.samples_csv_path}: {examples}"
        )
    return samples_df["__image_name__"].tolist()


def select_sample_row(corpus: RawCorpus, image_name: str) -> pd.Series:
    """Resolve one manifest row by exact image filename."""
    selected = str(image_name).strip()
    if not selected:
        raise ValueError("image_name cannot be blank.")

    samples_df = load_corpus_samples(corpus)
    matches = samples_df.loc[samples_df["__image_name__"] == selected]
    if matches.empty:
        raise FileNotFoundError(f"Image {selected!r} was not found in corpus {corpus.name}.")
    if len(matches) > 1:
        raise ValueError(
            f"Image {selected!r} matched multiple manifest rows in corpus {corpus.name}."
        )
    return matches.iloc[0].copy()
