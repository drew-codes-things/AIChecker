import array
import operator
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from ..sentences import split_sentences

CHUNK_TARGET_CHARS = 700
CHUNK_MAX_CHARS = 1100
CHUNK_MIN_CHARS = 120
EMBED_BATCH = 16
MAX_LIBRARY_CHUNKS = 3000
MIN_READY_CHUNKS = 3
MIN_READY_CHARS = 1500
NEAREST_SAMPLES = 4
MAX_PER_SOURCE = 2
PROFILE_SAMPLE_CHARS = 40_000
UPLOAD_ORIGIN = "upload"
FOLDER_ORIGIN = "folder"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    origin TEXT NOT NULL,
    path TEXT,
    signature TEXT,
    characters INTEGER NOT NULL,
    added REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_source ON chunks(source_id);
"""


class VoiceError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Chunk:
    id: int
    source_id: int
    text: str
    vector: array.array


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in text.replace("\r\n", "\n").split("\n\n") if part.strip()]


def _split_long(paragraph: str) -> list[str]:
    pieces, current = [], ""
    for span in split_sentences(paragraph):
        sentence = span.slice_of(paragraph)
        if current and len(current) + 1 + len(sentence) > CHUNK_MAX_CHARS:
            pieces.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return [piece[:CHUNK_MAX_CHARS] for piece in pieces]


def chunk_text(text: str) -> list[str]:
    units: list[str] = []
    for paragraph in _paragraphs(text):
        units.extend(_split_long(paragraph) if len(paragraph) > CHUNK_MAX_CHARS else [paragraph])
    chunks, current = [], ""
    for unit in units:
        if current and len(current) + 2 + len(unit) > CHUNK_TARGET_CHARS:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{unit}".strip()
    if current:
        chunks.append(current)
    kept = [chunk for chunk in chunks if len(chunk) >= CHUNK_MIN_CHARS]
    return kept or chunks[:1]


def evenly_spaced(items: Sequence, limit: int) -> list:
    if len(items) <= limit:
        return list(items)
    step = len(items) / limit
    return [items[int(index * step)] for index in range(limit)]


def normalise(vector: Sequence[float]) -> array.array:
    length = sqrt(sum(value * value for value in vector))
    if length == 0:
        raise VoiceError("the embedding came back as all zeros")
    return array.array("f", (value / length for value in vector))


def nearest(query: Sequence[float], chunks: Sequence[Chunk], count: int = NEAREST_SAMPLES,
            per_source: int = MAX_PER_SOURCE) -> list[Chunk]:
    unit_query = normalise(query)
    scored = sorted(((sum(map(operator.mul, unit_query, chunk.vector)), chunk) for chunk in chunks
                     if len(chunk.vector) == len(unit_query)), key=lambda pair: -pair[0])
    picked: list[Chunk] = []
    taken: dict[int, int] = {}
    for _, chunk in scored:
        if taken.get(chunk.source_id, 0) >= per_source:
            continue
        picked.append(chunk)
        taken[chunk.source_id] = taken.get(chunk.source_id, 0) + 1
        if len(picked) == count:
            break
    return picked


class VoiceLibrary:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self.clock = clock
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def add_source(self, name: str, origin: str, path: str | None, signature: str | None, characters: int,
                   chunks: Sequence[tuple[str, Sequence[float]]]) -> int:
        if not chunks:
            raise VoiceError("there was nothing to add")
        with self.lock, self._connect() as connection:
            existing = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if existing + len(chunks) > MAX_LIBRARY_CHUNKS:
                raise VoiceError(f"the voice library holds at most {MAX_LIBRARY_CHUNKS} passages; remove some files first", 409)
            cursor = connection.execute(
                "INSERT INTO sources (name, origin, path, signature, characters, added) VALUES (?, ?, ?, ?, ?, ?)",
                (name, origin, path, signature, characters, self.clock()))
            source_id = cursor.lastrowid
            connection.executemany(
                "INSERT INTO chunks (source_id, position, text, vector) VALUES (?, ?, ?, ?)",
                [(source_id, position, text, normalise(vector).tobytes()) for position, (text, vector) in enumerate(chunks)])
        return source_id

    def remove_source(self, source_id: int) -> bool:
        with self.lock, self._connect() as connection:
            return connection.execute("DELETE FROM sources WHERE id = ?", (source_id,)).rowcount > 0

    def sources(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT s.id, s.name, s.origin, s.characters, s.added, COUNT(c.id) AS passages "
                "FROM sources s LEFT JOIN chunks c ON c.source_id = s.id GROUP BY s.id ORDER BY s.added, s.id").fetchall()
        return [dict(row) for row in rows]

    def folder_signatures(self) -> dict[str, tuple[int, str | None]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id, path, signature FROM sources WHERE origin = ?", (FOLDER_ORIGIN,)).fetchall()
        return {row["path"]: (row["id"], row["signature"]) for row in rows}

    def chunks(self) -> list[Chunk]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id, source_id, text, vector FROM chunks ORDER BY id").fetchall()
        return [Chunk(row["id"], row["source_id"], row["text"], _unpack(row["vector"])) for row in rows]

    def totals(self) -> dict:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS passages, COALESCE(SUM(LENGTH(text)), 0) AS characters FROM chunks").fetchone()
            sources = connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        ready = row["passages"] >= MIN_READY_CHUNKS and row["characters"] >= MIN_READY_CHARS
        return {"sources": sources, "passages": row["passages"], "characters": row["characters"], "ready": ready}

    def profile_texts(self) -> list[str]:
        texts, used = [], 0
        for chunk in self.chunks():
            if used >= PROFILE_SAMPLE_CHARS:
                break
            texts.append(chunk.text)
            used += len(chunk.text)
        return texts


def _unpack(blob: bytes) -> array.array:
    vector = array.array("f")
    vector.frombytes(blob)
    return vector


def embed_chunks(chunks: Sequence[str], embed: Callable[[list[str]], list[list[float]]]) -> list[tuple[str, list[float]]]:
    embedded = []
    for start in range(0, len(chunks), EMBED_BATCH):
        batch = list(chunks[start:start + EMBED_BATCH])
        vectors = embed(batch)
        if len(vectors) != len(batch):
            raise VoiceError("the embedding service returned the wrong number of vectors", 502)
        embedded.extend(zip(batch, vectors))
    return embedded


def index_text(library: VoiceLibrary, name: str, text: str, embed: Callable[[list[str]], list[list[float]]],
               max_chunks: int, origin: str = UPLOAD_ORIGIN, path: str | None = None, signature: str | None = None) -> dict:
    chunks = chunk_text(text)
    if not chunks:
        raise VoiceError(f"{name} has no usable text")
    used = evenly_spaced(chunks, max_chunks)
    source_id = library.add_source(name, origin, path, signature, len(text), embed_chunks(used, embed))
    return {"id": source_id, "name": name, "passages": len(used), "passages_in_file": len(chunks)}
