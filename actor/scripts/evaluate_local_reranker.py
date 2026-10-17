"""Isolated ONNX CPU scorer; public weights only, no remote inference or code."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
PARENT = Path("artifacts/local-reranker-validation")
MODEL_DIR = PARENT / "model"
BASE = Path("artifacts/retrieval-reranker-validation/run-sq0gjqts")
CAPTURE = Path("artifacts/section-embedding-validation/run-eo81fta9")


def download(variant):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    sha = (
        json.loads((MODEL_DIR / "pin.json").read_text())["revision"]
        if variant == "arm64"
        else HfApi(token=False).model_info(MODEL).sha
    )
    files = {}
    model_file = "onnx/model_qint8_arm64.onnx" if variant == "arm64" else "onnx/model.onnx"
    for name in (model_file, "tokenizer.json", "config.json"):
        path = Path(hf_hub_download(MODEL, name, revision=sha, token=False, local_dir=MODEL_DIR))
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (MODEL_DIR / ("pin-arm64.json" if variant == "arm64" else "pin.json")).write_text(
        json.dumps({"repo": MODEL, "revision": sha, "sha256": files}, indent=2) + "\n"
    )
    print(json.dumps({"downloaded": MODEL, "revision": sha}), flush=True)


def main(variant):
    ort.disable_telemetry_events()
    pin = json.loads(
        (MODEL_DIR / ("pin-arm64.json" if variant == "arm64" else "pin.json")).read_text()
    )
    for name, expected in pin["sha256"].items():
        assert hashlib.sha256((MODEL_DIR / name).read_bytes()).hexdigest() == expected
    root = Path(tempfile.mkdtemp(prefix="run-", dir=PARENT))
    plan = {
        "model": pin,
        "variant": variant,
        "provider": "CPUExecutionProvider",
        "threads": 4,
        "max_pair_tokens": 512,
        "truncation": "only_second",
        "candidate_count": 16,
        "selected_count": 4,
        "profiles": ["full", "section", "max_whole_section"],
        "repeats": 3,
        "ranking": "Descending raw relevance logit; strongest representation per document for max profile. No fusion weights or labels used.",
        "gate": "No per-query recall4 regression vs IDs-only model baseline; warm selection p95 <=1 second. No default change.",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "dependencies": {
            n: importlib.metadata.version(n) for n in ("onnxruntime", "numpy", "tokenizers")
        },
    }
    (root / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    start = time.perf_counter()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(
            MODEL_DIR / ("onnx/model_qint8_arm64.onnx" if variant == "arm64" else "onnx/model.onnx")
        ),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=512, strategy="only_second")
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    load_seconds = time.perf_counter() - start
    input_names = {item.name for item in session.get_inputs()}

    def score(query, texts):
        start = time.perf_counter()
        encoded = tokenizer.encode_batch([(query, t) for t in texts])
        values = {
            "input_ids": np.asarray([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.asarray([e.attention_mask for e in encoded], dtype=np.int64),
            "token_type_ids": np.asarray([e.type_ids for e in encoded], dtype=np.int64),
        }
        scores = session.run(None, {k: values[k] for k in input_names})[0].reshape(-1).tolist()
        return scores, time.perf_counter() - start, sum(bool(e.overflowing) for e in encoded)

    _, warmup, _ = score("Find the delivery timing.", ["Delivery takes five business days."] * 16)
    cases = json.loads((CAPTURE / "manifest.json").read_text())["cases"]
    rows = []
    for case in cases:
        stem = case["suite"] + "--" + case["id"]
        req = json.loads((BASE / (stem + ".request.json")).read_text())
        docs = json.loads(req["messages"][1]["content"][0]["text"])["documents"]
        mapping = json.loads((BASE / (stem + ".mapping.json")).read_text())
        ids = [mapping[d["candidate_id"]]["entry_id"] for d in docs]
        sections = {
            r["entry_id"]: r["text"]
            for r in json.loads((CAPTURE / (stem + ".json")).read_text())["rankings"]["sections"]
        }
        full = [d["title"] + "\n\n" + d["body"] for d in docs]
        short = [sections[key] for key in ids]
        views = {"full": full, "section": short, "max_whole_section": full + short}
        for arm, texts in views.items():
            timings = []
            scores = None
            for _ in range(3):
                values, elapsed, truncated = score(case["query"], texts)
                current = (
                    values
                    if arm != "max_whole_section"
                    else [max(a, b) for a, b in zip(values[:16], values[16:], strict=True)]
                )
                if scores is not None:
                    assert np.allclose(scores, current)
                scores = current
                timings.append(elapsed)
            selected = sorted(ids, key=lambda key: (-scores[ids.index(key)], key))[:4]
            rows.append(
                {
                    "case": case["id"],
                    "arm": arm,
                    "selected_ids": selected,
                    "recall4": len(set(selected) & set(case["required"])) / len(case["required"]),
                    "timings_seconds": timings,
                    "median_seconds": statistics.median(timings),
                    "truncated_pairs": truncated,
                    "scores": dict(zip(ids, scores, strict=True)),
                }
            )
        print(json.dumps({"completed": case["id"], "rows": len(rows)}), flush=True)
        (root / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
    groups = {}
    for arm in views:
        chosen = [r for r in rows if r["arm"] == arm]
        times = [r["median_seconds"] for r in chosen]
        groups[arm] = {
            "mean_recall4": statistics.mean(r["recall4"] for r in chosen),
            "median_seconds": statistics.median(times),
            "p95_seconds": statistics.quantiles(times, n=100, method="inclusive")[94],
            "failed_cases": [r["case"] for r in chosen if r["recall4"] != 1.0],
            "truncated_pairs": sum(r["truncated_pairs"] for r in chosen),
        }
    result = {
        "groups": groups,
        "rows": rows,
        "load_seconds": load_seconds,
        "warmup_seconds": warmup,
        "default_changed": False,
        "remote_inference_calls": 0,
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(root)
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--variant", choices=("float32", "arm64"), default="float32")
    args = parser.parse_args()
    if args.download_only:
        download(args.variant)
    else:
        main(args.variant)
