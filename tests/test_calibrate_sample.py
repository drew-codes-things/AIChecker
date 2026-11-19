import csv
import json

from aichecker.worker.calibrate import build_sample, read_jsonl

FIELDS = ["id", "adv_source_id", "source_id", "model", "decoding", "repetition_penalty", "attack", "domain", "title", "prompt", "generation"]
TEXT = "A sentence that is long enough to be kept in the sample file for these tests. " * 4


def write_csv(path, sources=12, domains=("news", "poetry")):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for domain in domains:
            for number in range(sources):
                source = f"{domain}-{number}"
                writer.writerow({"id": source, "adv_source_id": source, "source_id": source, "model": "human", "decoding": "",
                                 "repetition_penalty": "", "attack": "none", "domain": domain, "title": "t", "prompt": "", "generation": TEXT})
                for model in ("gpt4", "mistral", "cohere"):
                    for decoding in ("greedy", "sampling"):
                        writer.writerow({"id": f"{source}-{model}-{decoding}", "adv_source_id": source, "source_id": source,
                                         "model": model, "decoding": decoding, "repetition_penalty": "no", "attack": "none",
                                         "domain": domain, "title": "t", "prompt": "p", "generation": TEXT + model})
                writer.writerow({"id": f"{source}-short", "adv_source_id": source, "source_id": source, "model": "gpt4", "decoding": "greedy",
                                 "repetition_penalty": "no", "attack": "none", "domain": domain, "title": "t", "prompt": "p", "generation": "short"})


def test_the_sample_has_one_human_and_one_ai_text_per_chosen_source_and_balanced_domains(tmp_path):
    source, out = tmp_path / "raid.csv", tmp_path / "sample.jsonl"
    write_csv(source)
    summary = build_sample(source, out, per_domain=5, seed=1)
    records = read_jsonl(out)
    assert summary == {"human": 10, "ai": 10, "sources": 10}
    per_source = {}
    for record in records:
        per_source.setdefault(record["source_id"], []).append(record["label"])
    assert all(sorted(labels) == [0, 1] for labels in per_source.values())
    assert {record["domain"] for record in records} == {"news", "poetry"}
    assert sum(1 for record in records if record["domain"] == "news") == 10


def test_the_same_seed_gives_the_same_sample_and_another_seed_a_different_one(tmp_path):
    source = tmp_path / "raid.csv"
    write_csv(source)
    build_sample(source, tmp_path / "a.jsonl", 5, 1)
    build_sample(source, tmp_path / "b.jsonl", 5, 1)
    build_sample(source, tmp_path / "c.jsonl", 5, 2)
    assert (tmp_path / "a.jsonl").read_text() == (tmp_path / "b.jsonl").read_text()
    assert (tmp_path / "a.jsonl").read_text() != (tmp_path / "c.jsonl").read_text()


def test_ai_rows_come_from_one_generator_per_source_and_too_short_texts_are_dropped(tmp_path):
    source, out = tmp_path / "raid.csv", tmp_path / "sample.jsonl"
    write_csv(source)
    build_sample(source, out, 12, 3)
    records = read_jsonl(out)
    assert all(len(record["text"]) >= 200 for record in records)
    assert not any(record["id"].endswith("-short") for record in records)
    models = {}
    for record in records:
        if record["label"] == 1:
            models.setdefault(record["source_id"], set()).add(record["model"])
    assert all(len(found) == 1 for found in models.values())


def test_long_texts_are_cut_for_the_sample_file(tmp_path):
    source, out = tmp_path / "raid.csv", tmp_path / "sample.jsonl"
    write_csv(source, sources=2)
    rows = list(csv.DictReader(open(source, encoding="utf-8")))
    for row in rows:
        row["generation"] = "x" * 9000
    with open(source, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    build_sample(source, out, 2, 1)
    assert all(len(json.loads(line)["text"]) == 4000 for line in open(out))
