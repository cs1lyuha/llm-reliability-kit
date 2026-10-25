from bench.reliability import CONFIGS, run


def test_each_layer_improves_reliability():
    results = {cfg: run(cfg) for cfg in CONFIGS}
    s = [results[c]["success"] for c in CONFIGS]
    assert s[0] < s[1] < s[2]
    assert s[3] >= s[2]


def test_circuit_breaker_cuts_wasted_calls_and_tail_latency():
    plain = run("retry + fallback")
    breaker = run("retry + fallback + circuit breaker")
    assert breaker["calls_per_req"] < plain["calls_per_req"]
    assert breaker["p95_latency"] < plain["p95_latency"]


def test_benchmark_is_deterministic():
    assert run("retry (4 încercări)") == run("retry (4 încercări)")
