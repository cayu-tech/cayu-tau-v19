import copy
import importlib.util
import sys
from pathlib import Path

import pytest

from cayu_tau3_banking.argument_contract import OPTIONAL_ARGUMENT_DEFAULTS, SOURCE_BINDING_RECOVERY


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location(
        "probe_binding_context", Path("scripts/probe_binding_context.py")
    )
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


def test_only_guidance_differs_and_source_is_untouched(monkeypatch):
    mod = module(monkeypatch)
    source = {
        "payload_redacted": False,
        "payload": {
            "model": "gpt-6-astra",
            "reasoning": {"effort": "high"},
            "instructions": "before\n" + OPTIONAL_ARGUMENT_DEFAULTS + "\nafter",
            "input": [
                {"role": "user", "content": "Source evidence"},
                {"type": "reasoning", "encrypted_content": "opaque"},
            ],
            "tools": [{"name": "test"}],
        },
    }
    original = copy.deepcopy(source)
    pair = mod.variants(source)
    a, b = pair["baseline"], pair["candidate"]
    assert a["model"] == b["model"] == "gpt-5.6-sol"
    assert a["reasoning"] == b["reasoning"] == {"effort": "xhigh"}
    assert a["input"] == b["input"] == [source["payload"]["input"][0]]
    assert b["instructions"].count(SOURCE_BINDING_RECOVERY) == 1
    b["instructions"] = a["instructions"]
    assert a == b
    assert source == original
    source["payload_redacted"] = True
    with pytest.raises(ValueError, match="complete captured"):
        mod.variants(source)
