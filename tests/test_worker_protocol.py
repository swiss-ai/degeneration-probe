"""Integration test for the inference worker WebSocket protocol."""

import asyncio
import json
import pytest

from degeneration_probe.worker.serve import handle_message


def test_handle_message_parses_generate_action():
    msg = json.dumps({
        "action": "generate",
        "prompt": "Hello",
        "params": {"temperature": 0.5, "top_p": 0.9, "max_new_tokens": 10},
        "steering": {"enabled": False},
    })
    parsed = json.loads(msg)
    assert parsed["action"] == "generate"
    assert parsed["prompt"] == "Hello"


def test_handle_message_parses_stop_action():
    msg = json.dumps({"action": "stop"})
    parsed = json.loads(msg)
    assert parsed["action"] == "stop"


def test_handle_message_parses_update_steering():
    msg = json.dumps({
        "action": "update_steering",
        "steering": {"enabled": True, "strategy": "temperature_boost", "threshold": 0.7},
    })
    parsed = json.loads(msg)
    assert parsed["action"] == "update_steering"
    assert parsed["steering"]["strategy"] == "temperature_boost"
