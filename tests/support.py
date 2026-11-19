from aichecker.calibration import FEATURE_NAMES, Calibration

HUMAN_POINTS = {
    "binoculars": [0.80 + 0.01 * index for index in range(21)],
    "fast_detect": [-3.0 + 0.1 * index for index in range(21)],
    "gltr_top1": [0.20 + 0.01 * index for index in range(21)],
    "gltr_top10": [0.50 + 0.01 * index for index in range(21)],
    "gltr_tail": [0.01 + 0.001 * index for index in range(21)],
}
AI_POINTS = {
    "binoculars": [0.60 + 0.01 * index for index in range(21)],
    "fast_detect": [0.0 + 0.2 * index for index in range(21)],
    "gltr_top1": [0.40 + 0.01 * index for index in range(21)],
    "gltr_top10": [0.70 + 0.01 * index for index in range(21)],
    "gltr_tail": [0.0 + 0.0005 * index for index in range(21)],
}


def make_calibration_data() -> dict:
    return {
        "schema": 1,
        "scorer": {"features": list(FEATURE_NAMES), "mean": [0.9, -1.0, 0.3, 0.6, 0.02],
                   "std": [0.1, 2.0, 0.1, 0.1, 0.01], "weights": [-1.5, 1.0, 0.4, 0.4, -0.3], "bias": -0.2},
        "thresholds": {"fpr1": 2.0, "fpr5": 1.0},
        "min_tokens": 48, "window_tokens": 512, "max_tokens": 4096,
        "sentence": {"min_tokens": 8, "strong": 0.7, "some": 0.85},
        "feature_quantiles": {name: {"human": HUMAN_POINTS[name], "ai": AI_POINTS[name]} for name in FEATURE_NAMES},
        "accuracy": {"overall": {"auroc": 0.9}},
        "pair": {"observer": "a", "performer": "b"},
        "created": "2026-11-18",
    }


def make_calibration() -> Calibration:
    return Calibration.from_dict(make_calibration_data())


HUMAN_LIKE = {"binoculars": 1.05, "fast_detect": -2.5, "gltr_top1": 0.22, "gltr_top10": 0.52, "gltr_tail": 0.02}
AI_LIKE = {"binoculars": 0.62, "fast_detect": 3.0, "gltr_top1": 0.55, "gltr_top10": 0.85, "gltr_tail": 0.001}


import io
import zipfile

WORD_HEAD = '<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
WORD_TAIL = "</w:body></w:document>"
ODT_HEAD = ('<?xml version="1.0" encoding="UTF-8"?><office:document-content '
            'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
            'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text>')
ODT_TAIL = "</office:text></office:body></office:document-content>"


def make_zip(members: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def make_docx(paragraphs: list[str], extra_body: str = "") -> bytes:
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    return make_zip({"word/document.xml": WORD_HEAD + body + extra_body + WORD_TAIL})


def make_odt(paragraphs: list[str]) -> bytes:
    body = "".join(f"<text:p>{text}</text:p>" for text in paragraphs)
    return make_zip({"content.xml": ODT_HEAD + body + ODT_TAIL})


def make_pdf(lines: list[str]) -> bytes:
    stream = "BT /F1 12 Tf 72 720 Td 14 TL " + " ".join(f"({line}) Tj T*" for line in lines) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
