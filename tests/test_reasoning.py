"""Reasoning normalization tests."""
from __future__ import annotations

import json

from app.reasoning import (
    StreamingInlineExtractor,
    StreamingReasoningNormalizer,
    normalize_message,
    normalize_response,
)


# --------------------------------------------------------------------------- #
# Non-streaming message normalization
# --------------------------------------------------------------------------- #


def test_normalize_renames_reasoning_field():
    msg = {"role": "assistant", "content": "hi", "reasoning": "thoughts"}
    out = normalize_message(msg)
    assert out["reasoning_content"] == "thoughts"
    assert "reasoning" not in out
    assert out["content"] == "hi"


def test_normalize_renames_thinking_field():
    msg = {"role": "assistant", "thinking": "hmm"}
    out = normalize_message(msg)
    assert out["reasoning_content"] == "hmm"
    assert "thinking" not in out


def test_normalize_renames_thought_field():
    msg = {"role": "assistant", "thought": "hmm"}
    out = normalize_message(msg)
    assert out["reasoning_content"] == "hmm"


def test_normalize_keeps_existing_reasoning_content():
    msg = {"role": "assistant", "reasoning_content": "rc", "reasoning": "extra"}
    out = normalize_message(msg)
    assert out["reasoning_content"] == "rcextra"


def test_normalize_inline_extraction():
    msg = {
        "role": "assistant",
        "content": "<think>internal thoughts</think>visible answer",
    }
    out = normalize_message(msg)
    assert out["reasoning_content"] == "internal thoughts"
    assert out["content"] == "visible answer"


def test_normalize_inline_extraction_multiple_blocks():
    msg = {
        "role": "assistant",
        "content": "<think>part one</think>visible1<think>part two</think>visible2",
    }
    out = normalize_message(msg)
    assert out["reasoning_content"] == "part one\npart two"
    assert out["content"] == "visible1visible2"


def test_normalize_combines_field_and_inline():
    msg = {
        "role": "assistant",
        "reasoning": "from field",
        "content": "<think>from inline</think>answer",
    }
    out = normalize_message(msg)
    # field reasoning first, inline appended (concatenated without separator)
    assert out["reasoning_content"] == "from fieldfrom inline"
    assert out["content"] == "answer"


def test_normalize_no_reasoning_unchanged():
    msg = {"role": "assistant", "content": "just text"}
    out = normalize_message(msg)
    assert out["content"] == "just text"
    assert "reasoning_content" not in out


def test_normalize_response_walks_choices():
    resp = {
        "id": "x",
        "choices": [
            {"message": {"role": "assistant", "reasoning": "r", "content": "c"}},
            {"message": {"role": "assistant", "content": "<think>t</think>c2"}},
        ],
    }
    out = normalize_response(resp)
    assert out["choices"][0]["message"]["reasoning_content"] == "r"
    assert out["choices"][1]["message"]["reasoning_content"] == "t"
    assert out["choices"][1]["message"]["content"] == "c2"


# --------------------------------------------------------------------------- #
# Streaming inline extractor (state machine)
# --------------------------------------------------------------------------- #


def test_extractor_simple_content():
    ex = StreamingInlineExtractor()
    out = ex.feed("hello world")
    out += ex.flush()
    assert out == [("content", "hello world")]


def test_extractor_full_think_in_one_chunk():
    ex = StreamingInlineExtractor()
    out = ex.feed("<think>thought</think>answer")
    out += ex.flush()
    assert out == [
        ("reasoning", "thought"),
        ("content", "answer"),
    ]


def test_extractor_split_open_tag():
    ex = StreamingInlineExtractor()
    out = []
    out += ex.feed("hello <th")
    out += ex.feed("ink>secret</think>final")
    out += ex.flush()
    contents = "".join(t for k, t in out if k == "content")
    reasoning = "".join(t for k, t in out if k == "reasoning")
    assert contents == "hello final"
    assert reasoning == "secret"


def test_extractor_split_close_tag():
    ex = StreamingInlineExtractor()
    out = []
    out += ex.feed("<think>part1</th")
    out += ex.feed("ink>part2")
    out += ex.flush()
    reasoning = "".join(t for k, t in out if k == "reasoning")
    content = "".join(t for k, t in out if k == "content")
    assert reasoning == "part1"
    assert content == "part2"


def test_extractor_partial_tag_at_end_of_buffer():
    """The last byte might be '<' which could be the start of <think>."""
    ex = StreamingInlineExtractor()
    out = []
    out += ex.feed("abc<")  # might be start of <think>
    # Without more input, we cannot be sure. The extractor should hold back.
    assert out == [("content", "abc")]
    out += ex.feed("def")  # actually was just a literal '<'
    out += ex.flush()
    assert "".join(t for k, t in out if k == "content") == "abc<def"


def test_extractor_nested_partial_prefixes():
    ex = StreamingInlineExtractor()
    out = []
    # Edge case: "<thi" could be start of <think> OR a literal
    out += ex.feed("<thi")
    assert out == []
    out += ex.feed("nking loudly")  # turns out it was literal "<thinking"
    out += ex.flush()
    content = "".join(t for k, t in out if k == "content")
    assert content == "<thinking loudly"


# --------------------------------------------------------------------------- #
# Streaming chunk normalizer
# --------------------------------------------------------------------------- #


def _parse_sse_payload(payload: str) -> dict:
    return json.loads(payload.removeprefix("data: ").strip())


def test_streaming_field_rename():
    norm = StreamingReasoningNormalizer()
    chunk = {
        "id": "x",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"reasoning": "thought"}, "finish_reason": None}],
    }
    out = norm.process_chunk(chunk)
    assert len(out) == 1
    delta = out[0]["choices"][0]["delta"]
    assert delta["reasoning_content"] == "thought"
    assert "reasoning" not in delta


def test_streaming_field_rename_thinking():
    norm = StreamingReasoningNormalizer()
    chunk = {"id": "x", "choices": [{"index": 0, "delta": {"thinking": "hmm"}}]}
    out = norm.process_chunk(chunk)
    assert out[0]["choices"][0]["delta"]["reasoning_content"] == "hmm"


def test_streaming_inline_extraction_in_one_chunk():
    norm = StreamingReasoningNormalizer()
    chunk = {
        "id": "x",
        "choices": [{"index": 0, "delta": {"content": "<think>r</think>c"}}],
    }
    out = norm.process_chunk(chunk)
    # Two output chunks: one reasoning, one content.
    assert len(out) == 2
    deltas = [c["choices"][0]["delta"] for c in out]
    assert deltas[0]["reasoning_content"] == "r"
    assert deltas[1]["content"] == "c"


def test_streaming_inline_extraction_across_chunks():
    norm = StreamingReasoningNormalizer()
    chunks = [
        {"id": "x", "choices": [{"index": 0, "delta": {"content": "<thi"}}]},
        {"id": "x", "choices": [{"index": 0, "delta": {"content": "nk>top secret"}}]},
        {"id": "x", "choices": [{"index": 0, "delta": {"content": "</think>answer"}}]},
    ]
    reasoning_parts = []
    content_parts = []
    for ch in chunks:
        for piece in norm.process_chunk(ch):
            delta = piece["choices"][0]["delta"]
            if "reasoning_content" in delta:
                reasoning_parts.append(delta["reasoning_content"])
            if "content" in delta:
                content_parts.append(delta["content"])
    assert "".join(reasoning_parts) == "top secret"
    assert "".join(content_parts) == "answer"


def test_streaming_passthrough_no_choices():
    """Some chunks (e.g. role-only first chunk) may have no content/reasoning."""
    norm = StreamingReasoningNormalizer()
    chunk = {"id": "x", "choices": [{"index": 0, "delta": {"role": "assistant"}}]}
    out = norm.process_chunk(chunk)
    assert len(out) == 1
    assert out[0]["choices"][0]["delta"] == {"role": "assistant"}


def test_streaming_preserves_finish_reason():
    norm = StreamingReasoningNormalizer()
    chunk = {
        "id": "x",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    out = norm.process_chunk(chunk)
    assert len(out) == 1
    assert out[0]["choices"][0]["finish_reason"] == "stop"


def test_streaming_preserves_top_level_fields():
    norm = StreamingReasoningNormalizer()
    chunk = {
        "id": "abc",
        "created": 12345,
        "model": "m",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "hi"}}],
    }
    out = norm.process_chunk(chunk)
    assert out[0]["id"] == "abc"
    assert out[0]["created"] == 12345
    assert out[0]["model"] == "m"
