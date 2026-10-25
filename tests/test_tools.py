from pydantic import BaseModel

from llmkit.providers.fake import ScriptedProvider, text
from llmkit.tools import ToolRegistry, run_tool_loop
from llmkit.types import LLMResponse, ToolCall


class OrderArgs(BaseModel):
    order_id: str


class RefundArgs(BaseModel):
    order_id: str
    amount: float


def make_registry():
    reg = ToolRegistry()
    refunds = []

    @reg.register("Caută o comandă după ID", OrderArgs)
    def get_order(order_id: str):
        if order_id == "X":
            raise KeyError("comanda nu există")
        return {"id": order_id, "status": "livrată", "total": 250.0}

    @reg.register("Rambursează o comandă", RefundArgs, needs_approval=True)
    def refund(order_id: str, amount: float):
        refunds.append((order_id, amount))
        return "rambursat"

    return reg, refunds


def tool_use(*calls):
    return LLMResponse(stop_reason="tool_use",
                       tool_calls=[ToolCall(f"t{i}", n, a) for i, (n, a) in enumerate(calls)])


def test_single_tool_round_trip():
    reg, _ = make_registry()
    p = ScriptedProvider([tool_use(("get_order", {"order_id": "A1"})), text("Comanda A1 e livrată.")])
    out = run_tool_loop(p, "unde e A1?", reg)
    assert out.text == "Comanda A1 e livrată."
    assert out.steps == 2
    tool_msg = p.requests[1].messages[-1]
    assert tool_msg["role"] == "user"
    assert tool_msg["content"][0]["tool_use_id"] == "t0"
    assert "livrată" in tool_msg["content"][0]["content"]


def test_parallel_calls_return_in_one_message():
    reg, _ = make_registry()
    p = ScriptedProvider([tool_use(("get_order", {"order_id": "A1"}), ("get_order", {"order_id": "A2"})),
                          text("ok")])
    run_tool_loop(p, "x", reg)
    results = p.requests[1].messages[-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t0", "t1"]


def test_tool_exception_becomes_error_result():
    reg, _ = make_registry()
    p = ScriptedProvider([tool_use(("get_order", {"order_id": "X"})), text("Nu găsesc comanda.")])
    out = run_tool_loop(p, "x", reg)
    assert out.events[0].is_error
    assert "KeyError" in out.events[0].output


def test_invalid_arguments_are_rejected_before_execution():
    reg, refunds = make_registry()
    p = ScriptedProvider([tool_use(("refund", {"order_id": "A1", "amount": "mult"})), text("ok")])
    out = run_tool_loop(p, "x", reg, approver=lambda c: True)
    assert out.events[0].is_error and "Argumente invalide" in out.events[0].output
    assert refunds == []


def test_unknown_tool_is_reported():
    reg, _ = make_registry()
    p = ScriptedProvider([tool_use(("delete_db", {})), text("ok")])
    out = run_tool_loop(p, "x", reg)
    assert "nu există" in out.events[0].output


def test_approval_required_and_denied():
    reg, refunds = make_registry()
    p = ScriptedProvider([tool_use(("refund", {"order_id": "A1", "amount": 10})), text("Am cerut aprobare.")])
    out = run_tool_loop(p, "x", reg)  # fără approver → respins
    assert refunds == []
    assert "respinsă" in out.events[0].output


def test_approval_granted_executes():
    reg, refunds = make_registry()
    p = ScriptedProvider([tool_use(("refund", {"order_id": "A1", "amount": 10})), text("Gata.")])
    run_tool_loop(p, "x", reg, approver=lambda c: c.input["amount"] < 100)
    assert refunds == [("A1", 10.0)]


def test_max_steps_stops_infinite_loop():
    reg, _ = make_registry()
    p = ScriptedProvider(default=lambda r: tool_use(("get_order", {"order_id": "A1"})))
    out = run_tool_loop(p, "x", reg, max_steps=3)
    assert out.stopped_reason == "max_steps"
    assert p.calls == 3


def test_tool_specs_are_sorted_and_strict():
    reg, _ = make_registry()
    specs = reg.specs()
    assert [s["name"] for s in specs] == ["get_order", "refund"]
    assert specs[1]["input_schema"]["additionalProperties"] is False
