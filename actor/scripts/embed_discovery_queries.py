"""Embed only a frozen query manifest and deterministic query facets, never documents."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path


async def run(args):
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    from cayu import TextEmbeddingRequest
    from cayu.providers import OpenAIProvider

    from cayu_tau3_banking.knowledge import _offer_facet_retrieval_text

    manifest = json.loads(args.cases.read_text())
    queries = [case["query"] for case in manifest["cases"]]
    texts = list(
        dict.fromkeys(
            queries + [facet for query in queries if (facet := _offer_facet_retrieval_text(query))]
        )
    )
    if not texts or len(texts) > 20 or sum(len(t.encode()) for t in texts) > 16000:
        raise ValueError("Frozen query-only payload exceeds limit")
    if args.output.exists():
        raise ValueError("Refusing to overwrite query vectors")
    provider = OpenAIProvider(api_key=args.key_file.read_text().strip(), timeout_s=60)
    try:
        result = await provider.embed_texts(
            TextEmbeddingRequest(model="text-embedding-3-large", dimensions=3072, texts=texts)
        )
        if result.model != "text-embedding-3-large" or sorted(
            e.index for e in result.embeddings
        ) != list(range(len(texts))):
            raise ValueError("Embedding response identity mismatch")
        payload = {
            "model": result.model,
            "dimensions": 3072,
            "case_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            "vectors": {texts[e.index]: e.vector for e in result.embeddings},
        }
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream)
        print(
            json.dumps(
                {
                    "embedded_queries": len(texts),
                    "payload_bytes": sum(len(t.encode()) for t in texts),
                    "usage": result.usage.model_dump(mode="json") if result.usage else None,
                }
            )
        )
    finally:
        await provider.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
