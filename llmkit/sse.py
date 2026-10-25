"""Server-Sent Events: codare pe server, parsare pe client, metrici de streaming.

Formatul e cel standard: blocuri `event: ...` / `data: ...` separate de o
linie goală. `data` e mereu JSON, ca textul cu newline-uri să nu strice
cadrul.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator


def encode(chunks: Iterable[str]) -> Iterator[str]:
    """Transformă bucățile de text în cadre SSE. Se termină cu `done` sau `error`."""
    try:
        for chunk in chunks:
            yield f"event: delta\ndata: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
    except Exception as err:
        yield f"event: error\ndata: {json.dumps({'error': str(err)}, ensure_ascii=False)}\n\n"
        return
    yield "event: done\ndata: {}\n\n"


def parse(lines: Iterable[str]) -> Iterator[tuple[str, dict]]:
    """Citește linii (de ex. dintr-un răspuns HTTP) și întoarce (eveniment, date)."""
    event, data = "message", []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith(":"):
            continue  # comentariu / keep-alive
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield event, json.loads("\n".join(data))


@dataclass
class StreamMetrics:
    time_to_first_token_s: float | None
    total_s: float
    chunks: int
    text: str


def measure(chunks: Iterable[str], clock: Callable[[], float] = time.perf_counter) -> StreamMetrics:
    start = clock()
    first = None
    parts: list[str] = []
    for chunk in chunks:
        if first is None:
            first = clock() - start
        parts.append(chunk)
    return StreamMetrics(first, clock() - start, len(parts), "".join(parts))
