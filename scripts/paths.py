"""Shared repository paths for data preparation scripts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = DATA_PROCESSED_DIR / "splits"
EMBEDDINGS_DIR = DATA_PROCESSED_DIR / "emb"
OUTPUTS_DIR = ROOT / "outputs"
CSV_DIR = OUTPUTS_DIR / "csv"
TXT_DIR = OUTPUTS_DIR / "txt"
JSON_DIR = OUTPUTS_DIR / "json"
MODELS_DIR = DATA_PROCESSED_DIR / "models"
SHOTSIZE_OUTPUT_DIR = MODELS_DIR / "finetune_shotsize"
SHOTSIZE_ENSEMBLE_DIR = OUTPUTS_DIR / "shotsize_ensemble"
FRAMING_OUTPUT_DIR = MODELS_DIR / "finetune_framing"


def report_path(directory: Path, name: str, root: Path = ROOT) -> Path:
    """Route reports under outputs by extension; leave custom directories unchanged."""
    outputs = (root / "outputs").resolve()
    directory = directory.resolve()
    try:
        relative = directory.relative_to(outputs)
    except ValueError:
        return directory / name
    category = {".csv": "csv", ".json": "json", ".txt": "txt"}.get(Path(name).suffix.lower())
    if category:
        parts = relative.parts
        if parts and parts[0] in ("csv", "json", "txt"):
            parts = parts[1:]
        return outputs / category / Path(*parts) / name
    return directory / name


def prepare_report(directory: Path, name: str, root: Path = ROOT) -> Path:
    path = report_path(directory, name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def split_file(name: str, root: Path = ROOT) -> Path:
    canonical = root / "data" / "processed" / "splits" / name
    legacy = root / "splits" / name
    return canonical if canonical.is_file() or not legacy.is_file() else legacy


def embedding_file(name: str, root: Path = ROOT) -> Path:
    canonical = root / "data" / "processed" / "emb" / name
    legacy = root / "emb" / name
    return canonical if canonical.is_file() or not legacy.is_file() else legacy


def shot_size_runs(root: Path = ROOT) -> list[Path]:
    model_dir = root / "data" / "processed" / "models"
    for folder in ("finetune_shotsize", "finetune_shotsize_protocol"):
        directory = model_dir / folder
        if (directory / "best_model.pt").is_file():
            return [directory]
        ensemble = sorted(p for p in directory.glob("seed*") if (p / "best_model.pt").is_file())
        if ensemble:
            return ensemble
    for base in (root / "outputs", root / "data" / "processed"):
        ensemble = sorted(
            path for path in (base / "finetune_shotsize_protocol").glob("seed*")
            if (path / "best_model.pt").is_file()
        )
        if ensemble:
            return ensemble
        single = base / "finetune_shotsize"
        if (single / "best_model.pt").is_file():
            return [single]
    return []


def framing_run(root: Path = ROOT) -> Path:
    canonical = root / "data" / "processed" / "models" / "finetune_framing" / "seed_42"
    legacy = root / "outputs" / "finetune_framing" / "seed_42"
    return legacy if not (canonical / "best_finetune.pt").is_file() and (legacy / "best_finetune.pt").is_file() else canonical


def ensemble_artifacts(root: Path = ROOT) -> tuple[Path, Path, Path]:
    """Return metrics, predictions directory, and probabilities directory."""
    logical = root / "outputs" / "shotsize_ensemble" / "ensemble"
    metrics = report_path(logical, "ensemble_metrics.json", root)
    if metrics.is_file():
        return (metrics, report_path(logical, "predictions_val_plain.csv", root).parent,
                root / "data" / "processed" / "shotsize_ensemble" / "ensemble")
    for legacy in (
        logical,
        root / "data" / "processed" / "models" / "finetune_shotsize" / "ensemble",
        root / "data" / "processed" / "models" / "finetune_shotsize_protocol" / "ensemble",
        root / "outputs" / "finetune_shotsize_protocol" / "ensemble",
        root / "data" / "processed" / "finetune_shotsize_protocol" / "ensemble",
    ):
        if (legacy / "ensemble_metrics.json").is_file():
            return legacy / "ensemble_metrics.json", legacy, legacy
    return (metrics, report_path(logical, "predictions_val_plain.csv", root).parent,
            root / "data" / "processed" / "shotsize_ensemble" / "ensemble")


def _input_dir(name: str) -> Path:
    for candidate in (DATA_RAW_DIR / name, ROOT / name):
        if candidate.exists():
            return candidate
    return DATA_RAW_DIR / name


SHOTQA_META_DIR = _input_dir("ShotQA_meta")
SHOTBENCH_DIR = _input_dir("ShotBench")
SHOTQA_IMAGES_DIR = _input_dir("data_shotqa")


def resolve_image_path(path: str | Path, root: Path = ROOT) -> Path:
    """Resolve existing image paths, including paths recorded before the data move."""
    original = root / path
    if original.is_file():
        return original
    parts = str(path).replace("\\", "/").split("/")
    for dataset in ("data_shotqa", "ShotBench"):
        indices = [i for i, part in enumerate(parts) if part == dataset]
        for index in reversed(indices):
            prefix = [part for part in parts[:index] if part and part != "."]
            is_project_path = not prefix or prefix[-1].casefold() == root.name.casefold()
            is_raw_path = len(prefix) >= 3 and prefix[-2:] == ["data", "raw"] and (
                prefix[-3].casefold() == root.name.casefold()
            )
            if (is_project_path or is_raw_path or prefix == ["data", "raw"]) and all(
                part not in ("", ".", "..") for part in parts[index + 1:]
            ):
                for base in (root / "data" / "raw", root):
                    candidate = base.joinpath(*parts[index:])
                    if candidate.is_file():
                        return candidate
    return original


def ensure_output_dirs() -> None:
    for directory in (CSV_DIR, TXT_DIR, JSON_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, SPLITS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
