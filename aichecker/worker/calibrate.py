import argparse
import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

from ..accuracy_text import render_markdown
from ..calibration import Calibration
from ..fit import FitError, build_calibration
from .engine import EngineLink
from .config import DEFAULT_ENGINE_URL, DEFAULT_LOCK_FILE, DEFAULT_MODEL_DIR
from .gate import GateBusy, GateSettings, GpuGate
from .records import MAX_TOKENS, WINDOW_TOKENS, ScoringInterrupted

log = logging.getLogger("aichecker.calibrate")

MAX_SAMPLE_CHARS = 4000
MIN_SAMPLE_CHARS = 200
GROUP_DOCS = 96
PAUSE_BETWEEN_GROUPS_SECONDS = 6.0
RETRY_WAIT_SECONDS = 30.0
SETTLE_WAIT_SECONDS = 1800.0
SETTLE_POLL_SECONDS = 5.0
GATE_WAIT_SECONDS = 900.0
MAX_CALIBRATION_IDLE_SECONDS = 3600.0
RAID_SOURCE = "RAID train_none.csv (no adversarial attacks)"


def read_rows(csv_path: Path):
    csv.field_size_limit(10**9)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def scan_catalogue(csv_path: Path) -> tuple[dict[str, list[str]], list[str]]:
    by_domain: dict[str, list[str]] = {}
    models: set[str] = set()
    for row in read_rows(csv_path):
        if row["model"] == "human":
            by_domain.setdefault(row["domain"], []).append(row["source_id"])
        else:
            models.add(row["model"])
    return by_domain, sorted(models)


def choose_sources(by_domain: dict[str, list[str]], per_domain: int, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    chosen = {}
    for domain, sources in sorted(by_domain.items()):
        for source in rng.sample(sources, min(per_domain, len(sources))):
            chosen[source] = domain
    return chosen


def build_sample(csv_path: Path, out_path: Path, per_domain: int, seed: int) -> dict:
    by_domain, models = scan_catalogue(csv_path)
    sources = choose_sources(by_domain, per_domain, seed)
    generator_rng = random.Random(seed + 1)
    generators = {source: generator_rng.choice(models) for source in sorted(sources)}
    rng = random.Random(seed + 2)
    picked_ai: dict[str, tuple[float, dict]] = {}
    humans = []
    for row in read_rows(csv_path):
        source = row["source_id"]
        if source not in sources or len(row["generation"]) < MIN_SAMPLE_CHARS:
            continue
        if row["model"] == "human":
            humans.append(row)
        elif row["model"] == generators[source]:
            ticket = rng.random()
            if source not in picked_ai or ticket < picked_ai[source][0]:
                picked_ai[source] = (ticket, row)
    records = [(row, 0) for row in humans] + [(entry[1], 1) for entry in picked_ai.values()]
    rng.shuffle(records)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row, label in records:
            handle.write(json.dumps({"id": row["id"], "source_id": row["source_id"], "label": label, "domain": row["domain"],
                                     "model": row["model"], "decoding": row["decoding"],
                                     "repetition_penalty": row["repetition_penalty"],
                                     "text": row["generation"][:MAX_SAMPLE_CHARS]}) + "\n")
    return {"human": len(humans), "ai": len(picked_ai), "sources": len(sources)}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_session(args):
    from .scorer import PairScorer
    from .server import free_gpu_mb
    from .session import GpuSession
    link = EngineLink(args.controller_url, args.engine_url)
    gate = GpuGate(GateSettings(Path(args.lock_file), 20.0, GATE_WAIT_SECONDS, 3.0, 9000), link, free_gpu_mb)
    scorer = PairScorer(Path(args.model_dir) / "Qwen2.5-1.5B", Path(args.model_dir) / "Qwen2.5-1.5B-Instruct")
    return GpuSession(gate, scorer, MAX_CALIBRATION_IDLE_SECONDS, GATE_WAIT_SECONDS + 60)


def score_group(session, group: list[dict]) -> list[dict]:
    with session.scoring() as scorer:
        documents = scorer.score_documents([record["text"] for record in group], session.should_yield)
    results = []
    for record, document in zip(group, documents):
        results.append({"id": record["id"], "source_id": record["source_id"], "label": record["label"],
                        "domain": record["domain"], "model": record["model"], "decoding": record["decoding"],
                        "repetition_penalty": record["repetition_penalty"], "tokens": document.tokens,
                        "features": document.features,
                        "sentences": [[stat.tokens, round(stat.nll, 4), round(stat.cross_entropy, 4)] for stat in document.sentences]})
    return results


def wait_until_settled(session) -> None:
    deadline = time.monotonic() + SETTLE_WAIT_SECONDS
    while not session.settle():
        if time.monotonic() >= deadline:
            log.warning("the model engine stayed busy for %.0f s; carrying on", SETTLE_WAIT_SECONDS)
            return
        time.sleep(SETTLE_POLL_SECONDS)


def run_scoring(args) -> int:
    sample = read_jsonl(Path(args.sample))
    out_path = Path(args.out)
    done = {record["id"] for record in read_jsonl(out_path)}
    todo = [record for record in sample if record["id"] not in done]
    log.info("%d documents in the sample, %d already scored, %d to go", len(sample), len(done), len(todo))
    session = build_session(args)
    started = time.monotonic()
    scored = 0
    position = 0
    while position < len(todo):
        group = todo[position:position + GROUP_DOCS]
        try:
            results = score_group(session, group)
        except (GateBusy, ScoringInterrupted) as busy:
            log.info("giving way (%s); trying again in %.0f s", busy, RETRY_WAIT_SECONDS)
            session.release()
            wait_until_settled(session)
            time.sleep(RETRY_WAIT_SECONDS)
            continue
        with open(out_path, "a", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result) + "\n")
        position += len(group)
        scored += len(group)
        session.release()
        wait_until_settled(session)
        elapsed = time.monotonic() - started
        log.info("scored %d/%d (%.2f docs/s)", position, len(todo), scored / elapsed)
        time.sleep(PAUSE_BETWEEN_GROUPS_SECONDS)
    session.stop()
    return 0


def fit_calibration(args) -> int:
    records = read_jsonl(Path(args.results))
    if not records:
        print(f"no results in {args.results}", file=sys.stderr)
        return 1
    pair = {"observer": "Qwen2.5-1.5B", "performer": "Qwen2.5-1.5B-Instruct", "dtype": "bfloat16"}
    notes = {"source": RAID_SOURCE, "documents": len(records), "truncation": f"first {WINDOW_TOKENS - 1} tokens of each text were scored in calibration",
             "split_seed": args.seed, "sample_note": args.note}
    try:
        calibration = build_calibration(records, args.seed, pair, notes, WINDOW_TOKENS, MAX_TOKENS)
    except FitError as error:
        print(f"could not fit: {error}", file=sys.stderr)
        return 1
    calibration.save(Path(args.out))
    Path(args.report).write_text(render_markdown(calibration.accuracy, calibration.created) + "\n", encoding="utf-8")
    print(json.dumps(calibration.accuracy["overall"], indent=2))
    return 0


def show_report(args) -> int:
    calibration = Calibration.load(Path(args.calibration))
    print(render_markdown(calibration.accuracy, calibration.created))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(prog="aichecker.worker.calibrate")
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--csv", required=True)
    sample.add_argument("--out", required=True)
    sample.add_argument("--per-domain", type=int, required=True)
    sample.add_argument("--seed", type=int, default=20261118)
    run = commands.add_parser("run")
    run.add_argument("--sample", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    run.add_argument("--lock-file", default=DEFAULT_LOCK_FILE)
    run.add_argument("--controller-url", default="")
    run.add_argument("--engine-url", default=DEFAULT_ENGINE_URL)
    fit = commands.add_parser("fit")
    fit.add_argument("--results", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--report", required=True)
    fit.add_argument("--seed", default="raid-split-1")
    fit.add_argument("--note", default="")
    report = commands.add_parser("report")
    report.add_argument("--calibration", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "sample":
        print(json.dumps(build_sample(Path(arguments.csv), Path(arguments.out), arguments.per_domain, arguments.seed)))
        return 0
    handlers = {"run": run_scoring, "fit": fit_calibration, "report": show_report}
    return handlers[arguments.command](arguments)


if __name__ == "__main__":
    sys.exit(main())
