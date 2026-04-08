"""Corpus discovery, manifest validation, shard inspection, and streaming loaders."""

from __future__ import annotations

import ast
from collections import OrderedDict
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np
import pandas as pd

from .paths import to_repo_relative
from .utils import sha256_file

REQUIRED_SAMPLES_COLUMNS = {
    "image_filename",
    "distance_m",
    "npz_filename",
    "npz_row_index",
}

REQUIRED_NPZ_KEYS = {"X", "y", "sample_id", "npz_row_index", "image_filename"}

ALLOWED_PADDING_MODES = {"disabled", "pad_to_max_bottom_right"}
ALLOWED_TRAIN_SHUFFLE_MODES = {"sequential", "shard", "active_shard_reservoir"}


@dataclass(frozen=True)
class CorpusInfo:
    """Authoritative manifest pointers and shard listing for one corpus."""

    source_root: str
    source_root_path: Path
    dataset_id: str
    corpus_dir: Path
    run_json_path: Path
    samples_csv_path: Path
    shard_paths: tuple[Path, ...]


@dataclass
class Batch:
    """Batch payload emitted by the streaming loader."""

    images: np.ndarray
    targets: np.ndarray
    rows: list[dict[str, Any]]


def discover_corpus_infos(data_root: Path, source_root: str) -> list[CorpusInfo]:
    """Discover corpuses under one source root with strict manifest authority checks."""
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {data_root}")

    corpuses = sorted(path for path in data_root.iterdir() if path.is_dir())
    if not corpuses:
        raise ValueError(f"No corpus directories found under {data_root}")

    infos: list[CorpusInfo] = []
    for corpus_dir in corpuses:
        run_candidates = sorted(corpus_dir.rglob("*run.json"))
        samples_candidates = sorted(corpus_dir.rglob("*samples.csv"))
        if len(run_candidates) != 1:
            raise ValueError(
                "Corpus must have exactly one authoritative *run.json; "
                f"found {len(run_candidates)} in {corpus_dir}: "
                + ", ".join(str(path) for path in run_candidates)
            )
        if len(samples_candidates) != 1:
            raise ValueError(
                "Corpus must have exactly one authoritative *samples.csv; "
                f"found {len(samples_candidates)} in {corpus_dir}: "
                + ", ".join(str(path) for path in samples_candidates)
            )

        shard_paths = tuple(sorted(corpus_dir.rglob("*_shard_*.npz")))
        if not shard_paths:
            raise ValueError(f"No shard files matching *_shard_*.npz found in {corpus_dir}")

        infos.append(
            CorpusInfo(
                source_root=source_root,
                source_root_path=data_root.resolve(),
                dataset_id=corpus_dir.name,
                corpus_dir=corpus_dir.resolve(),
                run_json_path=run_candidates[0].resolve(),
                samples_csv_path=samples_candidates[0].resolve(),
                shard_paths=shard_paths,
            )
        )
    return infos


def load_root_metadata(
    data_root: Path,
    source_root: str,
    repo_root: Path,
) -> tuple[pd.DataFrame, list[CorpusInfo]]:
    """Load authoritative manifests for one root into a combined dataframe."""
    infos = discover_corpus_infos(data_root, source_root=source_root)
    frames: list[pd.DataFrame] = []

    for info in infos:
        df = pd.read_csv(info.samples_csv_path)
        if df.empty:
            raise ValueError(f"Samples manifest is empty: {info.samples_csv_path}")
        missing_cols = sorted(REQUIRED_SAMPLES_COLUMNS - set(df.columns))
        if missing_cols:
            raise ValueError(
                f"Missing required columns in {info.samples_csv_path}: {missing_cols}"
            )

        distance_numeric = pd.to_numeric(df["distance_m"], errors="coerce")
        if distance_numeric.isna().any():
            bad = int(distance_numeric.isna().sum())
            raise ValueError(
                f"distance_m must be numeric in {info.samples_csv_path}; "
                f"found {bad} non-numeric rows."
            )
        df["distance_m"] = distance_numeric.astype(np.float32)

        npz_row_numeric = pd.to_numeric(df["npz_row_index"], errors="coerce")
        if npz_row_numeric.isna().any():
            bad = int(npz_row_numeric.isna().sum())
            raise ValueError(
                f"npz_row_index must be numeric in {info.samples_csv_path}; "
                f"found {bad} invalid rows."
            )
        df["npz_row_index"] = npz_row_numeric.astype(np.int64)
        df["npz_filename"] = df["npz_filename"].astype(str)

        df["source_root"] = info.source_root
        df["source_root_path"] = str(info.source_root_path)
        df["dataset_id"] = info.dataset_id
        df["corpus_dir"] = str(info.corpus_dir)
        df["run_json_path"] = str(info.run_json_path)
        df["samples_csv_path"] = str(info.samples_csv_path)
        df["relative_run_json_path"] = to_repo_relative(repo_root, info.run_json_path)
        df["relative_samples_csv_path"] = to_repo_relative(repo_root, info.samples_csv_path)
        df["relative_source_samples_csv_path"] = df["relative_samples_csv_path"]

        df["npz_path"] = df["npz_filename"].map(lambda name: str((info.corpus_dir / name).resolve()))
        missing_npz_mask = ~df["npz_path"].map(lambda value: Path(value).exists())
        if missing_npz_mask.any():
            sample_rows = df.loc[missing_npz_mask, ["npz_filename", "samples_csv_path"]].head(5)
            raise FileNotFoundError(
                "Found manifest rows pointing to missing shard files. "
                f"Examples:\n{sample_rows.to_string(index=False)}"
            )
        df["relative_npz_path"] = df["npz_path"].map(lambda value: to_repo_relative(repo_root, value))

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined, infos


def summarize_metadata(metadata_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize one root metadata dataframe."""
    if metadata_df.empty:
        raise ValueError("Cannot summarize empty metadata dataframe.")
    return {
        "source_root": str(metadata_df["source_root"].iloc[0]),
        "num_corpuses": int(metadata_df["dataset_id"].nunique()),
        "num_samples": int(len(metadata_df)),
        "num_unique_shards": int(metadata_df["npz_path"].nunique()),
        "dataset_ids": sorted(metadata_df["dataset_id"].unique().tolist()),
        "distance_m": {
            "min": float(metadata_df["distance_m"].min()),
            "max": float(metadata_df["distance_m"].max()),
            "mean": float(metadata_df["distance_m"].mean()),
            "std": float(metadata_df["distance_m"].std(ddof=0)),
        },
    }


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunk = stream.read(size)
    if len(chunk) != size:
        raise ValueError(f"Unexpected end-of-stream: expected {size} bytes, got {len(chunk)}")
    return chunk


def _read_npy_header_from_stream(stream: BinaryIO) -> tuple[np.dtype, tuple[int, ...], bool]:
    magic = _read_exact(stream, 6)
    if magic != b"\x93NUMPY":
        raise ValueError("Invalid NPY stream magic bytes.")
    major, minor = _read_exact(stream, 2)
    if major == 1:
        header_len = struct.unpack("<H", _read_exact(stream, 2))[0]
    elif major in (2, 3):
        header_len = struct.unpack("<I", _read_exact(stream, 4))[0]
    else:
        raise ValueError(f"Unsupported NPY version: {major}.{minor}")

    header_text = _read_exact(stream, header_len).decode("latin1").strip()
    metadata = ast.literal_eval(header_text)
    dtype = np.dtype(metadata["descr"])
    shape = tuple(metadata["shape"])
    fortran_order = bool(metadata["fortran_order"])
    return dtype, shape, fortran_order


def _read_npy_header_from_npz_member(npz_path: Path, member_name: str) -> tuple[np.dtype, tuple[int, ...], bool]:
    with zipfile.ZipFile(npz_path) as archive:
        with archive.open(member_name) as stream:
            return _read_npy_header_from_stream(stream)


def _infer_frame_geometry(x_shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Return (height, width, channels) for X-shape."""
    if len(x_shape) == 3:
        _, height, width = x_shape
        return int(height), int(width), 1
    if len(x_shape) == 4:
        n, a, b, c = x_shape
        if a in (1, 3, 4):
            return int(b), int(c), int(a)
        if c in (1, 3, 4):
            return int(a), int(b), int(c)
        raise ValueError(
            f"Unsupported 4D X layout {x_shape}; cannot infer channels safely."
        )
    raise ValueError(f"Unsupported X shape {x_shape}; expected 3D or 4D arrays.")


def inspect_shard_schema(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Inspect NPZ shard headers without decoding the full image tensor."""
    required_columns = {"dataset_id", "npz_path", "npz_row_index"}
    missing = sorted(required_columns - set(metadata_df.columns))
    if missing:
        raise ValueError(f"Missing required metadata columns for shard inspection: {missing}")

    rows: list[dict[str, Any]] = []
    grouped = metadata_df.groupby("npz_path", sort=True)
    for npz_path, group in grouped:
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Shard not found during inspection: {path}")
        with zipfile.ZipFile(path) as archive:
            member_names = [member.filename for member in archive.infolist() if member.filename.endswith(".npy")]
        keys = sorted(Path(name).stem for name in member_names)
        key_set = set(keys)
        x_dtype, x_shape, _ = _read_npy_header_from_npz_member(path, "X.npy")
        y_dtype, y_shape, _ = _read_npy_header_from_npz_member(path, "y.npy")
        sid_dtype, sid_shape, _ = _read_npy_header_from_npz_member(path, "sample_id.npy")
        ridx_dtype, ridx_shape, _ = _read_npy_header_from_npz_member(path, "npz_row_index.npy")
        height, width, channels = _infer_frame_geometry(tuple(int(v) for v in x_shape))

        rows.append(
            {
                "dataset_id": str(group["dataset_id"].iloc[0]),
                "npz_path": str(path.resolve()),
                "npz_filename": path.name,
                "keys": keys,
                "key_count": len(keys),
                "missing_required_keys": sorted(REQUIRED_NPZ_KEYS - key_set),
                "x_dtype": str(x_dtype),
                "x_shape": tuple(int(v) for v in x_shape),
                "y_dtype": str(y_dtype),
                "y_shape": tuple(int(v) for v in y_shape),
                "sample_id_dtype": str(sid_dtype),
                "sample_id_shape": tuple(int(v) for v in sid_shape),
                "npz_row_index_dtype": str(ridx_dtype),
                "npz_row_index_shape": tuple(int(v) for v in ridx_shape),
                "n_rows_from_x": int(x_shape[0]),
                "n_rows_manifest": int(len(group)),
                "row_index_min_manifest": int(group["npz_row_index"].min()),
                "row_index_max_manifest": int(group["npz_row_index"].max()),
                "height": int(height),
                "width": int(width),
                "channels": int(channels),
            }
        )

    schema_df = pd.DataFrame(rows).sort_values(["dataset_id", "npz_filename"]).reset_index(drop=True)
    if schema_df.empty:
        raise ValueError("Shard schema inspection produced no rows.")
    return schema_df


def validate_root_schema(metadata_df: pd.DataFrame, root_name: str) -> pd.DataFrame:
    """Validate manifest/shard schema linkage for one root."""
    if metadata_df.empty:
        raise ValueError(f"{root_name}: metadata is empty.")
    schema_df = inspect_shard_schema(metadata_df)

    if schema_df["missing_required_keys"].map(len).any():
        bad = schema_df[schema_df["missing_required_keys"].map(len) > 0]
        raise ValueError(
            f"{root_name}: shard(s) missing required NPZ keys:\n"
            + bad[["npz_filename", "missing_required_keys"]].to_string(index=False)
        )

    for record in schema_df.to_dict(orient="records"):
        npz_path = Path(record["npz_path"])
        n_x = int(record["n_rows_from_x"])
        n_manifest = int(record["n_rows_manifest"])
        y_shape = tuple(record["y_shape"])
        sid_shape = tuple(record["sample_id_shape"])
        row_shape = tuple(record["npz_row_index_shape"])

        if n_x != n_manifest:
            raise ValueError(
                f"{root_name}: shard row mismatch for {npz_path.name}; "
                f"X has {n_x}, manifest rows {n_manifest}."
            )
        if y_shape != (n_x,):
            raise ValueError(
                f"{root_name}: y shape mismatch in {npz_path.name}; expected {(n_x,)}, got {y_shape}."
            )
        if sid_shape != (n_x,):
            raise ValueError(
                f"{root_name}: sample_id shape mismatch in {npz_path.name}; expected {(n_x,)}, got {sid_shape}."
            )
        if row_shape != (n_x,):
            raise ValueError(
                f"{root_name}: npz_row_index shape mismatch in {npz_path.name}; expected {(n_x,)}, got {row_shape}."
            )

        shard_rows = metadata_df[metadata_df["npz_path"] == str(npz_path)].sort_values("npz_row_index")
        idx_values = shard_rows["npz_row_index"].to_numpy(dtype=np.int64)
        if np.unique(idx_values).shape[0] != idx_values.shape[0]:
            raise ValueError(f"{root_name}: duplicate npz_row_index values in manifest for {npz_path.name}.")
        if idx_values.min(initial=0) < 0 or idx_values.max(initial=-1) >= n_x:
            raise ValueError(
                f"{root_name}: npz_row_index out of range for {npz_path.name}; "
                f"expected [0, {n_x - 1}] got min={idx_values.min()} max={idx_values.max()}."
            )

        with np.load(npz_path, allow_pickle=False) as payload:
            shard_index = payload["npz_row_index"].astype(np.int64)
            shard_sample_id = payload["sample_id"].astype(str)
            shard_y = payload["y"].astype(np.float32)
            shard_img = payload["image_filename"].astype(str)

        expected_index = np.arange(n_x, dtype=np.int64)
        if not np.array_equal(shard_index, expected_index):
            raise ValueError(
                f"{root_name}: NPZ npz_row_index is not contiguous in {npz_path.name}."
            )

        csv_sample_id = shard_rows["sample_id"].astype(str).to_numpy()
        if csv_sample_id.shape[0] != shard_sample_id.shape[0] or not np.array_equal(
            csv_sample_id, shard_sample_id
        ):
            raise ValueError(
                f"{root_name}: sample_id alignment mismatch between samples.csv and {npz_path.name}."
            )

        if "image_filename" in shard_rows.columns:
            csv_image = shard_rows["image_filename"].astype(str).to_numpy()
            if csv_image.shape[0] != shard_img.shape[0] or not np.array_equal(csv_image, shard_img):
                raise ValueError(
                    f"{root_name}: image_filename alignment mismatch between samples.csv and {npz_path.name}."
                )

        csv_distance = shard_rows["distance_m"].to_numpy(dtype=np.float32)
        diff = float(np.max(np.abs(csv_distance - shard_y))) if n_x else 0.0
        if diff > 1e-5:
            raise ValueError(
                f"{root_name}: distance_m does not match shard y in {npz_path.name}; "
                f"max abs diff={diff}."
            )

    return schema_df


def determine_target_hw(schema_df: pd.DataFrame, padding_mode: str = "disabled") -> tuple[int, int]:
    """Determine frame geometry target based on schema and padding policy."""
    if padding_mode not in ALLOWED_PADDING_MODES:
        raise ValueError(
            f"Unsupported padding_mode={padding_mode}. Allowed: {sorted(ALLOWED_PADDING_MODES)}"
        )
    shapes = sorted({(int(h), int(w)) for h, w in zip(schema_df["height"], schema_df["width"])})
    if len(shapes) == 1:
        return shapes[0]
    if padding_mode == "disabled":
        raise ValueError(
            "Inconsistent frame geometry detected and padding_mode is disabled. "
            f"Observed (height, width): {shapes}. "
            "Enable padding_mode='pad_to_max_bottom_right' to pad without scaling."
        )
    max_h = max(h for h, _ in shapes)
    max_w = max(w for _, w in shapes)
    return max_h, max_w


def detect_overlap_warnings(
    train_metadata: pd.DataFrame,
    val_metadata: pd.DataFrame,
    check_shard_hashes: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    """Detect leakage-risk overlap signals between training and validation roots."""
    warnings: list[str] = []
    details: dict[str, Any] = {}

    train_samples_paths = set(train_metadata["relative_samples_csv_path"].astype(str))
    val_samples_paths = set(val_metadata["relative_samples_csv_path"].astype(str))
    shared_manifest_paths = sorted(train_samples_paths & val_samples_paths)
    if shared_manifest_paths:
        warnings.append(
            "Training and validation reference identical relative samples.csv paths."
        )
        details["shared_relative_samples_paths"] = shared_manifest_paths

    train_run_paths = set(train_metadata["relative_run_json_path"].astype(str))
    val_run_paths = set(val_metadata["relative_run_json_path"].astype(str))
    shared_run_paths = sorted(train_run_paths & val_run_paths)
    if shared_run_paths:
        warnings.append("Training and validation reference identical relative run.json paths.")
        details["shared_relative_run_paths"] = shared_run_paths

    train_sample_ids = set(train_metadata["sample_id"].astype(str))
    val_sample_ids = set(val_metadata["sample_id"].astype(str))
    shared_sample_ids = sorted(train_sample_ids & val_sample_ids)
    if shared_sample_ids:
        warnings.append("Training and validation contain overlapping sample_id values.")
        details["overlapping_sample_id_count"] = len(shared_sample_ids)
        details["overlapping_sample_id_examples"] = shared_sample_ids[:20]

    train_pairs = set(
        zip(
            train_metadata["dataset_id"].astype(str),
            train_metadata["sample_id"].astype(str),
        )
    )
    val_pairs = set(
        zip(
            val_metadata["dataset_id"].astype(str),
            val_metadata["sample_id"].astype(str),
        )
    )
    shared_pairs = sorted(train_pairs & val_pairs)
    if shared_pairs:
        warnings.append("Training and validation share (dataset_id, sample_id) pairs.")
        details["overlapping_dataset_sample_pairs"] = shared_pairs[:20]
        details["overlapping_dataset_sample_pair_count"] = len(shared_pairs)

    if check_shard_hashes:
        train_shards = sorted({Path(path) for path in train_metadata["npz_path"].astype(str)})
        val_shards = sorted({Path(path) for path in val_metadata["npz_path"].astype(str)})
        train_hashes = {path.name: sha256_file(path) for path in train_shards}
        val_hashes = {path.name: sha256_file(path) for path in val_shards}
        hash_to_train = {digest: name for name, digest in train_hashes.items()}
        hash_to_val = {digest: name for name, digest in val_hashes.items()}
        shared_hashes = sorted(set(hash_to_train) & set(hash_to_val))
        if shared_hashes:
            warnings.append("Training and validation contain identical shard file hashes.")
            details["shared_shard_hash_matches"] = [
                {
                    "train_shard": hash_to_train[digest],
                    "val_shard": hash_to_val[digest],
                    "sha256": digest,
                }
                for digest in shared_hashes
            ]

    return warnings, details


class NpyRowStream:
    """Sequential row streamer for an array stored inside an NPZ archive."""

    def __init__(self, npz_path: Path, array_key: str = "X") -> None:
        self.npz_path = Path(npz_path)
        self.array_key = array_key
        self._archive: zipfile.ZipFile | None = None
        self._stream: BinaryIO | None = None
        self.dtype: np.dtype | None = None
        self.shape: tuple[int, ...] | None = None
        self.row_shape: tuple[int, ...] | None = None
        self.row_bytes: int | None = None

    def __enter__(self) -> "NpyRowStream":
        self._archive = zipfile.ZipFile(self.npz_path)
        self._stream = self._archive.open(f"{self.array_key}.npy")
        dtype, shape, fortran_order = _read_npy_header_from_stream(self._stream)
        if fortran_order:
            raise ValueError(
                f"Unsupported Fortran-order array in {self.npz_path}:{self.array_key}.npy"
            )
        if len(shape) < 2:
            raise ValueError(
                f"Expected at least 2 dimensions for {self.npz_path}:{self.array_key}.npy; got {shape}"
            )
        self.dtype = dtype
        self.shape = tuple(int(v) for v in shape)
        self.row_shape = tuple(int(v) for v in shape[1:])
        self.row_bytes = int(np.prod(self.row_shape)) * int(dtype.itemsize)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._stream is not None:
            self._stream.close()
        if self._archive is not None:
            self._archive.close()
        self._stream = None
        self._archive = None

    def iter_rows(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (row_index, ndarray_row) sequentially."""
        if self._stream is None or self.shape is None or self.dtype is None or self.row_shape is None:
            raise RuntimeError("NpyRowStream must be opened with a context manager before iter_rows().")
        if self.row_bytes is None:
            raise RuntimeError("row_bytes not initialized.")

        for row_index in range(int(self.shape[0])):
            raw = _read_exact(self._stream, self.row_bytes)
            row = np.frombuffer(raw, dtype=self.dtype).reshape(self.row_shape)
            yield row_index, row


class ShardArrayCache:
    """Memory-budgeted LRU cache for decoded shard `X` arrays."""

    def __init__(self, max_bytes: int, name: str) -> None:
        self.max_bytes = int(max(0, max_bytes))
        self.name = str(name)
        self._entries: OrderedDict[str, np.ndarray] = OrderedDict()
        self._bytes_used = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.skipped_too_large = 0

    @property
    def bytes_used(self) -> int:
        return int(self._bytes_used)

    def _key(self, npz_path: Path) -> str:
        return str(npz_path.resolve())

    def _required_x_bytes(self, npz_path: Path) -> int:
        dtype, shape, _ = _read_npy_header_from_npz_member(npz_path, "X.npy")
        return int(np.prod(shape)) * int(dtype.itemsize)

    def _load_x_array(self, npz_path: Path) -> np.ndarray:
        with np.load(npz_path, allow_pickle=False) as payload:
            return np.asarray(payload["X"])

    def _evict_until_fits(self, required_bytes: int) -> None:
        while self._entries and (self._bytes_used + required_bytes) > self.max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self._bytes_used -= int(evicted.nbytes)
            self.evictions += 1

    def get_or_load_x(
        self,
        npz_path: Path,
        allow_temporary_load: bool = False,
    ) -> np.ndarray | None:
        """Return cached shard X array, or load it if policy allows."""
        key = self._key(npz_path)
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key, last=True)
            self.hits += 1
            return cached

        if self.max_bytes <= 0:
            return self._load_x_array(npz_path) if allow_temporary_load else None

        required_bytes = self._required_x_bytes(npz_path)
        if required_bytes > self.max_bytes:
            self.skipped_too_large += 1
            return self._load_x_array(npz_path) if allow_temporary_load else None

        self.misses += 1
        loaded = self._load_x_array(npz_path)
        self._evict_until_fits(required_bytes=int(loaded.nbytes))
        self._entries[key] = loaded
        self._bytes_used += int(loaded.nbytes)
        return loaded

    def preload(self, shard_paths: list[Path] | tuple[Path, ...]) -> None:
        for shard_path in shard_paths:
            _ = self.get_or_load_x(Path(shard_path), allow_temporary_load=False)

    def stats(self) -> dict[str, Any]:
        total_requests = int(self.hits + self.misses)
        hit_rate = float(self.hits / total_requests) if total_requests else 0.0
        return {
            "name": self.name,
            "max_bytes": int(self.max_bytes),
            "bytes_used": int(self._bytes_used),
            "cached_shards": int(len(self._entries)),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "evictions": int(self.evictions),
            "skipped_too_large": int(self.skipped_too_large),
            "hit_rate": float(hit_rate),
        }


def _to_grayscale_2d(image: np.ndarray) -> np.ndarray:
    """Normalize image layout into 2D grayscale without geometry changes."""
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        # Channel-first if first axis looks like channels.
        if image.shape[0] in (1, 3, 4):
            channels_first = image
            if channels_first.shape[0] == 1:
                gray = channels_first[0]
            else:
                rgb = channels_first[:3].astype(np.float32)
                gray = 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]
        # Channel-last if last axis looks like channels.
        elif image.shape[-1] in (1, 3, 4):
            channels_last = image
            if channels_last.shape[-1] == 1:
                gray = channels_last[..., 0]
            else:
                rgb = channels_last[..., :3].astype(np.float32)
                gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
        else:
            raise ValueError(f"Cannot infer channel axis for image shape {image.shape}.")
    else:
        raise ValueError(f"Unsupported image ndim={image.ndim}; expected 2 or 3 dimensions.")

    gray = np.asarray(gray)
    if gray.dtype == np.uint8:
        out = gray.astype(np.float32) / 255.0
    elif np.issubdtype(gray.dtype, np.integer):
        max_value = float(np.iinfo(gray.dtype).max)
        out = gray.astype(np.float32) / max_value
    else:
        out = gray.astype(np.float32)
        if out.min(initial=0.0) < 0.0 or out.max(initial=1.0) > 1.0:
            raise ValueError(
                "Float image values must already be in [0, 1] for this first-pass pipeline."
            )
    return out


def _pad_to_target(image_2d: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    h, w = image_2d.shape
    if h > target_h or w > target_w:
        raise ValueError(
            f"Cannot pad image of shape {(h, w)} into smaller target {(target_h, target_w)}."
        )
    if h == target_h and w == target_w:
        return image_2d
    out = np.zeros((target_h, target_w), dtype=image_2d.dtype)
    out[:h, :w] = image_2d
    return out


def prepare_image_for_model(
    image: np.ndarray,
    target_hw: tuple[int, int],
    padding_mode: str = "disabled",
) -> np.ndarray:
    """Convert one sample image into float32 [1, H, W] for model input."""
    if padding_mode not in ALLOWED_PADDING_MODES:
        raise ValueError(
            f"Unsupported padding_mode={padding_mode}. Allowed: {sorted(ALLOWED_PADDING_MODES)}"
        )
    gray = _to_grayscale_2d(image)
    h, w = gray.shape
    if (h, w) != target_hw:
        if padding_mode == "pad_to_max_bottom_right":
            gray = _pad_to_target(gray, target_hw=target_hw)
        else:
            raise ValueError(
                "Image geometry mismatch with padding disabled. "
                f"Image {(h, w)} vs expected {target_hw}."
            )
    return gray[None, ...].astype(np.float32)


def iter_batches(
    metadata_df: pd.DataFrame,
    batch_size: int,
    target_hw: tuple[int, int],
    padding_mode: str = "disabled",
    shuffle_shards: bool = False,
    seed: int = 42,
    shard_cache: ShardArrayCache | None = None,
    shuffle_mode: str | None = None,
    active_shard_count: int = 3,
) -> Iterator[Batch]:
    """Stream mini-batches from shard-backed metadata.

    shuffle_mode:
    - sequential: deterministic shard/path order.
    - shard: shuffle shard order each epoch, keep within-shard row order.
    - active_shard_reservoir: keep N shards active and sample rows from them randomly.
    """
    if metadata_df.empty:
        raise ValueError("Cannot iterate batches from empty metadata.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}")
    if shuffle_mode is None:
        shuffle_mode = "shard" if shuffle_shards else "sequential"
    if shuffle_mode not in ALLOWED_TRAIN_SHUFFLE_MODES:
        raise ValueError(
            f"Unsupported shuffle_mode={shuffle_mode}; expected one of {sorted(ALLOWED_TRAIN_SHUFFLE_MODES)}"
        )
    if active_shard_count <= 0:
        raise ValueError(f"active_shard_count must be positive; got {active_shard_count}")

    grouped = {
        str(path): group.sort_values("npz_row_index").reset_index(drop=True)
        for path, group in metadata_df.groupby("npz_path", sort=False)
    }
    shard_paths = sorted(grouped.keys())
    rng = np.random.default_rng(seed)

    image_buffer: list[np.ndarray] = []
    target_buffer: list[float] = []
    row_buffer: list[dict[str, Any]] = []

    def _emit_if_ready() -> Iterator[Batch]:
        if len(image_buffer) >= batch_size:
            yield Batch(
                images=np.stack(image_buffer).astype(np.float32),
                targets=np.asarray(target_buffer, dtype=np.float32),
                rows=list(row_buffer),
            )
            image_buffer.clear()
            target_buffer.clear()
            row_buffer.clear()

    def _append_sample(image: np.ndarray, row_record: dict[str, Any]) -> Iterator[Batch]:
        model_image = prepare_image_for_model(
            image=image,
            target_hw=target_hw,
            padding_mode=padding_mode,
        )
        image_buffer.append(model_image)
        target_buffer.append(float(row_record["distance_m"]))
        row_buffer.append(row_record)
        yield from _emit_if_ready()

    if shuffle_mode == "active_shard_reservoir":
        if shard_cache is None:
            raise ValueError(
                "shuffle_mode='active_shard_reservoir' requires a shard_cache "
                "so rows can be sampled non-sequentially."
            )

        active_target = min(int(active_shard_count), len(shard_paths))
        pending_shards = list(shard_paths)
        rng.shuffle(pending_shards)
        active_states: list[dict[str, Any]] = []

        def _activate_next_shard() -> None:
            while pending_shards and len(active_states) < active_target:
                shard_path = pending_shards.pop()
                shard_rows = grouped[shard_path]
                if shard_rows.empty:
                    continue
                shard_x = shard_cache.get_or_load_x(
                    Path(shard_path), allow_temporary_load=True
                )
                if shard_x is None:
                    raise RuntimeError(f"Failed to load shard array for {shard_path}")
                max_required = int(shard_rows["npz_row_index"].max())
                if int(shard_x.shape[0]) <= max_required:
                    raise ValueError(
                        f"Requested row {max_required} exceeds shard rows "
                        f"{int(shard_x.shape[0])} for {shard_path}."
                    )
                row_positions = rng.permutation(len(shard_rows))
                active_states.append(
                    {
                        "shard_path": shard_path,
                        "rows": shard_rows,
                        "row_positions": row_positions,
                        "pointer": 0,
                        "x": shard_x,
                    }
                )

        _activate_next_shard()

        while active_states:
            state_index = int(rng.integers(0, len(active_states)))
            state = active_states[state_index]
            pointer = int(state["pointer"])
            row_positions = state["row_positions"]
            rows_df = state["rows"]
            row_pos = int(row_positions[pointer])
            state["pointer"] = pointer + 1

            row_record = rows_df.iloc[row_pos].to_dict()
            row_index = int(row_record["npz_row_index"])
            image = state["x"][row_index]
            yield from _append_sample(image=image, row_record=row_record)

            if int(state["pointer"]) >= len(row_positions):
                active_states.pop(state_index)
                _activate_next_shard()

        if image_buffer:
            yield Batch(
                images=np.stack(image_buffer).astype(np.float32),
                targets=np.asarray(target_buffer, dtype=np.float32),
                rows=list(row_buffer),
            )
        return

    if shuffle_mode == "shard":
        rng.shuffle(shard_paths)

    for shard_path in shard_paths:
        shard_rows = grouped[shard_path]
        requested = shard_rows["npz_row_index"].to_numpy(dtype=np.int64)
        if requested.size == 0:
            continue
        request_pointer = 0
        last_required_index = int(requested[-1])

        cached_x: np.ndarray | None = None
        if shard_cache is not None:
            cached_x = shard_cache.get_or_load_x(
                Path(shard_path), allow_temporary_load=False
            )

        if cached_x is not None:
            if int(cached_x.shape[0]) <= last_required_index:
                raise ValueError(
                    f"Requested row {last_required_index} exceeds cached shard rows "
                    f"{int(cached_x.shape[0])} for {shard_path}."
                )
            while request_pointer < requested.size:
                row_index = int(requested[request_pointer])
                image = cached_x[row_index]
                row_record = shard_rows.iloc[request_pointer].to_dict()
                yield from _append_sample(image=image, row_record=row_record)
                request_pointer += 1
        else:
            with NpyRowStream(Path(shard_path), array_key="X") as stream:
                for row_index, image in stream.iter_rows():
                    if row_index > last_required_index:
                        break
                    if row_index != int(requested[request_pointer]):
                        continue

                    row_record = shard_rows.iloc[request_pointer].to_dict()
                    yield from _append_sample(image=image, row_record=row_record)
                    request_pointer += 1

                    if request_pointer >= requested.size:
                        break

        if request_pointer != requested.size:
            missing = requested[request_pointer:]
            preview = ", ".join(str(int(v)) for v in missing[:8])
            raise ValueError(
                f"Failed to stream all requested rows from {shard_path}; "
                f"missing npz_row_index values starting with [{preview}]"
            )

    if image_buffer:
        yield Batch(
            images=np.stack(image_buffer).astype(np.float32),
            targets=np.asarray(target_buffer, dtype=np.float32),
            rows=list(row_buffer),
        )


def read_image_preview(npz_path: Path, npz_row_index: int) -> np.ndarray:
    """Read one raw image sample for notebook inspection."""
    target_index = int(npz_row_index)
    if target_index < 0:
        raise ValueError(f"npz_row_index must be non-negative; got {target_index}")
    with NpyRowStream(npz_path, array_key="X") as stream:
        for idx, image in stream.iter_rows():
            if idx == target_index:
                return np.array(image)
            if idx > target_index:
                break
    raise IndexError(f"Row index {target_index} out of range for shard {npz_path}")
