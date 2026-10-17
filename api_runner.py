#!/usr/bin/env python3
"""Run the frozen campaign supervisor with the actor on an OpenAI API key.

Used by `reproduce.py live-check|run` for workspaces set up with
`--actor-provider api`. The frozen scripts/run_official_campaign.py probe always
builds the Codex subscription provider, so this replaces only that connectivity
probe. Workers, checkpoints, trial pause, recovery and scoring are the unchanged
frozen supervisor; workers build the actor provider from protocol.json, which
setup changed to `--provider api`.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import aclosing
import importlib.util
import os
from pathlib import Path


def load_campaign(actor):
    path = actor / 'scripts/run_official_campaign.py'
    spec = importlib.util.spec_from_file_location('run_official_campaign', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def api_probe(campaign):
    async def probe(root, key_file):
        from cayu import Message
        from cayu.providers.base import ModelRequest, ModelStreamEventType
        from openai import AsyncOpenAI

        cli, protocol = campaign.install_runtime(root, key_file)
        arguments = protocol['cli_arguments']
        if arguments[arguments.index('--provider') + 1] != 'api':
            raise ValueError('Workspace was not set up with --actor-provider api.')
        model = arguments[arguments.index('--model') + 1]
        effort = arguments[arguments.index('--reasoning-effort') + 1]
        provider = cli.build_provider(provider='api', codex_lb_base_url='',
                                      semantic_progress_timeout_seconds=45)
        try:
            async with asyncio.timeout(90):
                completed = False
                request = ModelRequest(model=model, messages=[Message.text('user', 'Reply OK.')],
                                       options={'thinking': {'enabled': True, 'effort': effort}})
                async with aclosing(provider.runtime_stream(request)) as stream:
                    async for event in stream:
                        if event.type is ModelStreamEventType.ERROR:
                            raise RuntimeError('Actor API connectivity failed')
                        completed |= event.type is ModelStreamEventType.COMPLETED
                if not completed:
                    raise RuntimeError('Actor API completion missing')
            # Simulator and embedding checks are the same as the frozen probe.
            async with AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'], timeout=60, max_retries=0) as api:
                response = await api.chat.completions.create(model='gpt-5.2',
                    messages=[{'role': 'user', 'content': 'Reply OK.'}], reasoning_effort='low',
                    max_completion_tokens=128)
                if not response.choices:
                    raise RuntimeError('Simulator connectivity failed')
                vectors = await api.embeddings.create(model='text-embedding-3-large',
                                                      input='connectivity', dimensions=3072)
                if len(vectors.data[0].embedding) != 3072:
                    raise RuntimeError('Embedding connectivity failed')
        finally:
            await provider.aclose()
    return probe


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_known_args()[0].root.resolve()
    campaign = load_campaign(root)
    # supervise() and the probe command look up this module-level name at call time.
    campaign.probe = api_probe(campaign)
    campaign.main()


if __name__ == '__main__':
    main()
