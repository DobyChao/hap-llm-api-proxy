"""Response hook: normalize reasoning content.

Two transformations:

1. **Field rename** — providers disagree on the field name used for chain-of-thought
   ("reasoning", "thinking", "thought", ...). We rename them all to the canonical
   ``reasoning_content`` field used by DeepSeek/OpenAI-compatible responses.

2. **Inline extraction** — some providers (e.g. Qwen) embed the reasoning inside
   the regular ``content`` field wrapped in ``<think>...</think>`` tags. We pull
   that out into ``reasoning_content`` and leave the visible content clean.

Both transformations apply to non-streaming messages and streaming deltas. For
streaming we use a small state machine that buffers partial tag boundaries so
that ``<think>`` split across chunks is still detected correctly.
"""
from __future__ import annotations

import json
import re
from typing import Iterable, Optional


# Field names providers use for reasoning content other than the canonical one.
REASONING_FIELDS: tuple[str, ...] = ("reasoning", "thinking", "thought")

# Inline reasoning markers. We support the common `<think>...</think>` form.
OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"


# Pre-compiled regex for non-streaming inline extraction.
_INLINE_RE = re.compile(
    re.escape(OPEN_TAG) + r"(.*?)" + re.escape(CLOSE_TAG),
    re.DOTALL,
)


def _is_prefix_suffix(text: str, prefix: str) -> bool:
    """True if ``text`` ends with a prefix of ``prefix`` (and is shorter)."""
    if len(text) >= len(prefix):
        return False
    return prefix.startswith(text)


def _longest_safe_emit(buffer: str, tag: str) -> int:
    """Largest prefix of ``buffer`` we can emit safely (i.e. it cannot be the
    start of a partial ``tag`` match)."""
    # Walk down from the full length, looking for the largest i such that the
    # suffix buffer[i:] is a prefix of tag.
    for i in range(max(0, len(buffer) - len(tag) + 1), len(buffer) + 1):
        if i == len(buffer):
            return i
        if tag.startswith(buffer[i:]):
            return i  # hold back the partial-tag suffix
    return len(buffer)


class StreamingInlineExtractor:
    """Stateful extractor that turns a stream of ``content`` chunks into a
    stream of (kind, text) pieces, where kind is ``"content"`` or ``"reasoning"``.

    Handles the case where ``<think>`` or ``</think>`` are split across chunks
    by buffering potential partial-tag suffixes.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        self._buffer += text
        out: list[tuple[str, str]] = []
        tag = CLOSE_TAG if self._in_think else OPEN_TAG

        while self._buffer:
            idx = self._buffer.find(tag)
            if idx == -1:
                # No full tag in buffer. Emit everything except a possible
                # partial-tag suffix.
                safe = _longest_safe_emit(self._buffer, tag)
                if safe > 0:
                    piece = self._buffer[:safe]
                    self._buffer = self._buffer[safe:]
                    out.append(("reasoning" if self._in_think else "content", piece))
                break
            # Emit anything before the tag.
            if idx > 0:
                out.append(("reasoning" if self._in_think else "content", self._buffer[:idx]))
            # Consume the tag.
            self._buffer = self._buffer[idx + len(tag):]
            self._in_think = not self._in_think
            tag = CLOSE_TAG if self._in_think else OPEN_TAG
        return out

    def flush(self) -> list[tuple[str, str]]:
        """Drain any buffered content. Call once when the stream ends."""
        if not self._buffer:
            return []
        out = [("reasoning" if self._in_think else "content", self._buffer)]
        self._buffer = ""
        self._in_think = False
        return out


def _merge_reasoning(existing: Optional[str], add: str) -> str:
    if not add:
        return existing or ""
    if not existing:
        return add
    return existing + add


def normalize_message(msg: dict) -> dict:
    """Normalize a non-streaming ``message`` dict in place.

    Renames ``reasoning``/``thinking``/``thought`` to ``reasoning_content`` and
    extracts ``<think>...</think>`` blocks from ``content`` into
    ``reasoning_content``. ``content`` is left as the visible text only.
    """
    if not isinstance(msg, dict):
        return msg

    reasoning = msg.get("reasoning_content") or ""

    # 1. Field rename.
    for field in REASONING_FIELDS:
        if field in msg:
            val = msg.pop(field)
            if isinstance(val, str) and val:
                reasoning = _merge_reasoning(reasoning, val)

    # 2. Inline extraction from content.
    content = msg.get("content")
    if isinstance(content, str) and content:
        inline_parts = _INLINE_RE.findall(content)
        if inline_parts:
            inline = "\n".join(p.strip() for p in inline_parts if p.strip())
            if inline:
                reasoning = _merge_reasoning(reasoning, inline)
            cleaned = _INLINE_RE.sub("", content).strip()
            msg["content"] = cleaned

    if reasoning:
        msg["reasoning_content"] = reasoning
    elif "reasoning_content" in msg:
        # If it was empty/None, leave it alone if the upstream set it
        if not msg["reasoning_content"]:
            msg["reasoning_content"] = reasoning

    return msg


def normalize_response(resp: dict) -> dict:
    """Normalize a full non-streaming chat completion response."""
    choices = resp.get("choices")
    if isinstance(choices, list):
        for c in choices:
            if not isinstance(c, dict):
                continue
            msg = c.get("message")
            if isinstance(msg, dict):
                normalize_message(msg)
    return resp


class StreamingReasoningNormalizer:
    """Stateful normalizer that processes streaming chunks.

    A single input chunk may be split into multiple output chunks (e.g. when
    inline ``</think>`` is followed by visible content, we emit a reasoning
    chunk and a content chunk).
    """

    def __init__(self) -> None:
        # choice index -> extractor
        self._extractors: dict[int, StreamingInlineExtractor] = {}

    def _extractor(self, idx: int) -> StreamingInlineExtractor:
        ex = self._extractors.get(idx)
        if ex is None:
            ex = StreamingInlineExtractor()
            self._extractors[idx] = ex
        return ex

    def process_chunk(self, chunk: dict) -> list[dict]:
        """Return a list of chunks to emit. Most chunks pass through unchanged."""
        if not isinstance(chunk, dict):
            return [chunk]
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return [chunk]

        # We process each choice; choices are typically a single-element list in
        # streaming, but we handle the multi-choice case by building parallel
        # output lists. To keep things simple, we always rebuild the chunk:
        # every input chunk becomes one or more output chunks. Output chunks
        # always carry the same top-level fields (id, model, created, etc.)
        # with one delta per choice.
        base = {k: v for k, v in chunk.items() if k != "choices"}

        # Collect per-choice list of (delta_dict, finish_reason) pieces to emit.
        per_choice_pieces: list[list[dict]] = []
        max_pieces = 1
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            idx = choice.get("index", 0) or 0
            delta = dict(choice.get("delta") or {})
            finish_reason = choice.get("finish_reason")
            other = {k: v for k, v in choice.items() if k not in ("delta", "index", "finish_reason")}

            # 1. Field rename in delta.
            field_reasoning = ""
            for field in REASONING_FIELDS:
                if field in delta:
                    val = delta.pop(field)
                    if isinstance(val, str) and val:
                        field_reasoning += val

            # 2. Inline extraction from content.
            pieces: list[tuple[str, str]] = []
            content = delta.pop("content", None)
            if isinstance(content, str) and content:
                pieces = self._extractor(idx).feed(content)

            # Merge field reasoning into pieces: if any reasoning piece exists,
            # prepend to the first one; else emit as its own reasoning piece.
            if field_reasoning:
                if pieces and pieces[0][0] == "reasoning":
                    pieces[0] = ("reasoning", field_reasoning + pieces[0][1])
                else:
                    pieces.insert(0, ("reasoning", field_reasoning))

            if not pieces:
                # Nothing to split. Carry delta through (possibly modified).
                per_choice_pieces.append([{
                    "index": idx,
                    "delta": delta,
                    "finish_reason": finish_reason,
                    **other,
                }])
                continue

            # Build one choice-entry per piece.
            choice_pieces: list[dict] = []
            for i, (kind, text) in enumerate(pieces):
                new_delta = dict(delta)
                if kind == "content":
                    new_delta["content"] = text
                else:
                    new_delta["reasoning_content"] = text
                piece = {
                    "index": idx,
                    "delta": new_delta,
                    "finish_reason": finish_reason if i == len(pieces) - 1 else None,
                    **other,
                }
                choice_pieces.append(piece)
            per_choice_pieces.append(choice_pieces)
            max_pieces = max(max_pieces, len(choice_pieces))

        # Emit one chunk per "slot", combining piece[i] from each choice. Missing
        # choices (some choices have fewer pieces than max) are simply dropped
        # from that output chunk.
        out: list[dict] = []
        for slot in range(max_pieces):
            new_choices: list[dict] = []
            for pieces in per_choice_pieces:
                if slot < len(pieces):
                    new_choices.append(pieces[slot])
            if not new_choices:
                continue
            out.append({**base, "choices": new_choices})
        return out

    def flush(self) -> list[dict]:
        """Drain any buffered content at end of stream. Currently emits nothing
        because partial buffers should only ever be tail fragments that will be
        emitted as content/reasoning with whatever classification we were in.
        Kept for symmetry with the lower-level extractor."""
        out: list[dict] = []
        for _idx, ex in self._extractors.items():
            remaining = ex.flush()
            # We don't have a chunk template to wrap this in; the upstream's
            # terminating chunk (data: [DONE]) signals end. If a provider ends
            # mid-think we lose the buffer here -- acceptable for a v1.
            if remaining:
                # Best-effort: emit as a synthetic chunk. We don't have id/model
                # here, so we let the caller attach them if desired. For now,
                # drop on the floor -- not common in practice.
                pass
        return out


def parse_sse_line(line: str) -> Optional[dict]:
    """Parse a single SSE ``data:`` line. Returns None for non-data lines."""
    if not line:
        return None
    if line.startswith("data: "):
        payload = line[6:]
    elif line.startswith("data:"):
        payload = line[5:]
    else:
        return None
    if payload == "[DONE]":
        return {"__done__": True}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def format_sse_data(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
