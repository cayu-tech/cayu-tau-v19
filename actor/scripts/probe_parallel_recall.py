"""Live retrieval concurrency qualification using generic public-policy queries."""

import argparse
import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cayu.providers import OpenAIProvider  # noqa: E402
from evaluate_section_embeddings import CachedEmbeddings  # noqa: E402
from run_full_benchmark import verify, write_json  # noqa: E402

from cayu_tau3_banking.cli import build_memory_corpus  # noqa: E402
from cayu_tau3_banking.runtime_embeddings import OwnedEmbeddingProvider  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openai-api-key-file", type=Path, required=True)
    args = parser.parse_args()
    verify(ROOT)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    key = args.openai_api_key_file.read_text().strip()
    with OwnedEmbeddingProvider(lambda: OpenAIProvider(api_key=key)) as provider:
        cache = CachedEmbeddings(provider)
        path = Path(protocol["corpus_embedding_cache"])
        metadata = json.loads((path / "manifest.json").read_text())
        cache.restore(path, metadata["corpus_sha256"])
        corpus = asyncio.run(build_memory_corpus(
            Path(protocol["tau_repository"]) / "data/tau2/domains/banking_knowledge/documents",
            retrieval_mode="hybrid", embedding_model="text-embedding-3-large",
            embedding_dimensions=3072, embedding_provider=cache))
        def query(text):
            started = time.monotonic()
            try:
                result = asyncio.run(corpus.recall(text))
                value = {"passed": True, "candidate_count": result.candidate_count,
                         "source_coverage": result.source_coverage}
            except Exception as error:
                value = {"passed": False, "error_type": type(error).__name__,
                         "source": getattr(error, "source", None), "code": getattr(error, "code", None)}
            return {**value, "seconds": time.monotonic() - started}
        single = query("card replacement")
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            parallel = list(pool.map(query, ["purchase dispute", "account opening", "close card",
                "fee waiver", "lost card", "transfer limit", "interest rate", "cash withdrawal"]))
        passed = single["passed"] and all(row["passed"] for row in parallel)
        write_json(ROOT / "parallel-recall-qualification.json", {"passed": passed,
            "purpose": "Generic public-query concurrency check, no benchmark task or scoring",
            "single": single, "parallel": parallel, "parallel_wall_seconds": time.monotonic() - started})
        print({"passed": passed, "parallel_successes": sum(row["passed"] for row in parallel)})
        raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
