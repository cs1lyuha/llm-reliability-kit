from llmkit import sse


def test_round_trip_keeps_newlines_and_unicode():
    chunks = ["Bună ", "ziua,\nlinia doi", " ✓"]
    frames = "".join(sse.encode(chunks)).splitlines(keepends=True)
    events = list(sse.parse(frames))
    assert [e for e, _ in events] == ["delta", "delta", "delta", "done"]
    assert "".join(d["text"] for e, d in events if e == "delta") == "".join(chunks)


def test_error_in_stream_becomes_error_event():
    def broken():
        yield "a"
        raise RuntimeError("provider căzut")

    events = list(sse.parse("".join(sse.encode(broken())).splitlines()))
    assert events[-1] == ("error", {"error": "provider căzut"})


def test_comments_and_keepalive_are_ignored():
    lines = [": ping", "", "event: delta", 'data: {"text": "x"}', ""]
    assert list(sse.parse(lines)) == [("delta", {"text": "x"})]


def test_measure_time_to_first_token():
    ticks = iter([0.0, 0.4, 1.0])
    m = sse.measure(["a", "b"], clock=lambda: next(ticks))
    assert m.time_to_first_token_s == 0.4
    assert m.total_s == 1.0
    assert m.text == "ab"
