from collections.abc import Mapping


def _percent(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _number(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _row(name: str, entry: Mapping) -> str:
    return (f"| {name} | {entry.get('n_human', '')} | {entry.get('n_ai', '')} | {_number(entry.get('auroc'))} | "
            f"{_percent(entry.get('tpr_at_1pct_fpr'))} | {_percent(entry.get('tpr_at_5pct_fpr'))} | "
            f"{_percent(entry.get('fpr_at_fpr1_threshold'))} | {_percent(entry.get('tpr_at_fpr1_threshold'))} | "
            f"{_percent(entry.get('fpr_at_fpr5_threshold'))} | {_percent(entry.get('tpr_at_fpr5_threshold'))} |")


HEADER = ("| group | human | AI | AUROC | TPR at 1% FPR | TPR at 5% FPR | FPR at strong cut | TPR at strong cut | "
          "FPR at some cut | TPR at some cut |\n|---|---|---|---|---|---|---|---|---|---|")
GENERATOR_HEADER = ("| generator | AI | AUROC vs all human | TPR at strong cut | TPR at some cut |\n|---|---|---|---|---|")


def _generator_row(name: str, entry: Mapping) -> str:
    return (f"| {name} | {entry.get('n_ai', '')} | {_number(entry.get('auroc'))} | "
            f"{_percent(entry.get('tpr_at_fpr1_threshold'))} | {_percent(entry.get('tpr_at_fpr5_threshold'))} |")


def render_markdown(accuracy: Mapping, created: str) -> str:
    lines = [f"Measured {created} on held-out RAID documents (never used to fit anything).", ""]
    choice = accuracy["scorer_choice"]
    lines += [f"Signals used: {', '.join(choice['chosen'])}. Chosen on a separate validation split; "
              f"candidates scored {choice['candidates_validation_auroc']}.", ""]
    split = accuracy["split"]
    lines += [f"Documents: train {split['train']}, validation {split['validation']}, test {split['test']}. "
              "Splits are by source text, so no passage appears on both sides.", ""]
    lines += ["### Overall (test split)", "", HEADER, _row("all", accuracy["overall"]), ""]
    lines += ["### By domain", "", HEADER] + [_row(name, entry) for name, entry in accuracy["per_domain"].items()] + [""]
    lines += ["### By text length (tokens)", "", HEADER] + [_row(name, entry) for name, entry in accuracy["per_length"].items()] + [""]
    lines += ["### By generator", "", GENERATOR_HEADER] + [_generator_row(name, entry) for name, entry in accuracy["per_generator"].items()] + [""]
    lines += ["### By decoding", "", GENERATOR_HEADER] + [_generator_row(name, entry) for name, entry in accuracy["per_decoding"].items()] + [""]
    sentence = accuracy["sentence_level"]
    lines += ["### Sentence marks", "",
              f"AUROC of a single sentence's predictability against its document's label: {_number(sentence['auroc'])}. "
              f"Strong mark: {_percent(sentence['ai_sentences_marked_strong'])} of AI sentences, "
              f"{_percent(sentence['human_sentences_marked_strong'])} of human sentences. "
              f"Some mark: {_percent(sentence['ai_sentences_marked_some'])} of AI sentences, "
              f"{_percent(sentence['human_sentences_marked_some'])} of human sentences.", ""]
    return "\n".join(lines)
