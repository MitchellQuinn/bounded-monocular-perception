"""Discovery helpers for inference models and raw-image corpora."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .paths import input_root as default_input_root
from .paths import models_root as default_models_root
from .paths import repo_root

_TRUE_VALUES = {"1", "true", "t", "yes", "y"}


@dataclass(frozen=True)
class ModelRunArtifact:
    """One selectable trained run artifact."""

    label: str
    model_family: str
    model_name: str
    run_id: str
    model_dir: Path
    run_dir: Path
    config_path: Path
    metadata_path: Path | None
    checkpoint_path: Path


@dataclass(frozen=True)
class RawCorpus:
    """One selectable raw-image corpus staged under the inference input contract."""

    name: str
    root: Path
    images_dir: Path
    run_json_path: Path
    samples_csv_path: Path


def normalize_model_family(value: str | None) -> str:
    """Normalize one model-family identifier."""
    text = str(value or "").strip().lower()
    if not text:
        return ""
    aliases = {
        "distance": "distance-orientation",
        "distance orientation": "distance-orientation",
        "distance-orientation": "distance-orientation",
        "distance_orientation": "distance-orientation",
        "regression": "distance-orientation",
        "roi": "roi-fcn",
        "roi-fcn": "roi-fcn",
        "roi_fcn": "roi-fcn",
        "roifcn": "roi-fcn",
    }
    return aliases.get(text, text)


def default_raw_corpus_roots() -> list[Path]:
    """Return the ordered raw-corpus roots searched by default."""
    candidates = [default_input_root()]
    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(resolved)
    return roots


def _discover_config_paths(models_root: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in ("**/runs/run_*/config.json", "**/runs/run_*/run_config.json"):
        for config_path in sorted(models_root.glob(pattern)):
            resolved = config_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield resolved


def _infer_model_family(models_root: Path, *, model_dir: Path, config_path: Path) -> str:
    try:
        relative_parts = model_dir.relative_to(models_root).parts
    except ValueError:
        relative_parts = ()

    if len(relative_parts) >= 2:
        inferred = normalize_model_family(relative_parts[0])
        if inferred:
            return inferred
    if len(relative_parts) == 1:
        inferred = normalize_model_family(models_root.name)
        if inferred:
            return inferred

    if config_path.name == "run_config.json":
        return "roi-fcn"
    if config_path.name == "config.json":
        return "distance-orientation"
    return "unknown"


def _resolve_metadata_path(run_dir: Path) -> Path | None:
    for filename in ("run_manifest.json", "dataset_contract.json", "dataset_summary.json"):
        candidate = run_dir / filename
        if candidate.is_file():
            return candidate.resolve()
    return None


def discover_model_runs(
    root: Path | None = None,
    *,
    family: str | None = None,
) -> list[ModelRunArtifact]:
    """Discover runnable model artifacts under `./models`."""
    models_root = (root or default_models_root()).resolve()
    if not models_root.exists():
        raise FileNotFoundError(f"Models root does not exist: {models_root}")

    selected_family = normalize_model_family(family)
    artifacts: list[ModelRunArtifact] = []
    for config_path in _discover_config_paths(models_root):
        run_dir = config_path.parent.resolve()
        model_dir = run_dir.parent.parent.resolve()
        model_family = _infer_model_family(models_root, model_dir=model_dir, config_path=config_path)
        if selected_family and model_family != selected_family:
            continue

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

        artifacts.append(
            ModelRunArtifact(
                label=f"{model_family} / {model_dir.name} / {run_dir.name}",
                model_family=model_family,
                model_name=model_dir.name,
                run_id=run_dir.name,
                model_dir=model_dir,
                run_dir=run_dir,
                config_path=config_path,
                metadata_path=_resolve_metadata_path(run_dir),
                checkpoint_path=checkpoint_path.resolve(),
            )
        )
    return sorted(
        artifacts,
        key=lambda artifact: (
            artifact.model_family,
            artifact.model_name,
            artifact.run_id,
        ),
    )


def _discover_raw_corpora_under_root(input_root: Path) -> list[RawCorpus]:
    """Discover raw-image corpora under one concrete root."""
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


def discover_raw_corpora(root: Path | None = None) -> list[RawCorpus]:
    """Discover raw-image corpora.

    When `root` is omitted, search the local inference `input/` folder.
    """
    if root is not None:
        return _discover_raw_corpora_under_root(Path(root).resolve())

    checked_roots = default_raw_corpus_roots()
    existing_roots = [candidate for candidate in checked_roots if candidate.exists()]
    if not existing_roots:
        joined = ", ".join(str(path) for path in checked_roots)
        raise FileNotFoundError(f"No raw-corpus roots exist. Checked: {joined}")

    corpora: list[RawCorpus] = []
    seen_roots: set[Path] = set()
    for candidate_root in existing_roots:
        for corpus in _discover_raw_corpora_under_root(candidate_root):
            if corpus.root in seen_roots:
                continue
            seen_roots.add(corpus.root)
            corpora.append(corpus)
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
