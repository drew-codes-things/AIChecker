import os
from collections.abc import Callable
from pathlib import Path

from .extract import MAX_UPLOAD_BYTES, SUPPORTED_EXTENSIONS, ExtractError, extract_text
from .voice import FOLDER_ORIGIN, VoiceError, VoiceLibrary, index_text

MAX_FOLDER_FILES = 300
MAX_FOLDER_DEPTH = 3
MAX_FILE_CHARACTERS = 400_000


def list_voice_files(folder: Path) -> list[Path]:
    root = Path(folder).resolve()
    if not root.is_dir():
        raise VoiceError(f"the voice folder {root} does not exist or is not a folder", 404)
    found: list[Path] = []
    for directory, subdirectories, names in os.walk(root, followlinks=False):
        depth = len(Path(directory).relative_to(root).parts)
        if depth >= MAX_FOLDER_DEPTH:
            subdirectories[:] = []
        for name in sorted(names):
            candidate = Path(directory) / name
            if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS or candidate.is_symlink():
                continue
            resolved = candidate.resolve()
            if root not in resolved.parents:
                continue
            found.append(resolved)
            if len(found) >= MAX_FOLDER_FILES:
                return found
    return found


def signature_of(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def rescan_folder(library: VoiceLibrary, folder: Path, embed: Callable[[list[str]], list[list[float]]],
                  max_chunks: int) -> dict:
    files = list_voice_files(folder)
    known = library.folder_signatures()
    present = {str(path) for path in files}
    removed = 0
    for path, (source_id, _) in known.items():
        if path not in present and library.remove_source(source_id):
            removed += 1
    added, unchanged, skipped = [], 0, []
    for path in files:
        signature = signature_of(path)
        previous = known.get(str(path))
        if previous and previous[1] == signature:
            unchanged += 1
            continue
        if previous:
            library.remove_source(previous[0])
        try:
            if path.stat().st_size > MAX_UPLOAD_BYTES:
                raise ExtractError(f"larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB", 413)
            extracted = extract_text(path.name, path.read_bytes(), MAX_FILE_CHARACTERS)
            added.append(index_text(library, path.name, extracted.text, embed, max_chunks, FOLDER_ORIGIN, str(path), signature))
        except (ExtractError, VoiceError, OSError) as error:
            skipped.append({"name": path.name, "reason": str(error)})
    return {"added": len(added), "unchanged": unchanged, "removed": removed, "skipped": skipped}
