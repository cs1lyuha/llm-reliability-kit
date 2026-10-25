import json
from typing import Literal

import pytest
from pydantic import BaseModel, Field, field_validator

from llmkit.providers.fake import ScriptedProvider, text
from llmkit.structured import StructuredOutputError, extract, strict_schema


class Address(BaseModel):
    city: str
    street: str


class Order(BaseModel):
    customer: str
    quantity: int = Field(ge=1)
    priority: Literal["low", "normal", "urgent"]
    address: Address

    @field_validator("customer")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("numele clientului lipsește")
        return v


GOOD = {"customer": "Ion Rusu", "quantity": 3, "priority": "urgent",
        "address": {"city": "Chișinău", "street": "Ștefan cel Mare 1"}}


def test_valid_json_first_try():
    p = ScriptedProvider([text(json.dumps(GOOD))])
    out = extract(p, "comanda", Order)
    assert out.value.address.city == "Chișinău"
    assert out.repairs == 0


def test_schema_is_sent_and_closed():
    p = ScriptedProvider([text(json.dumps(GOOD))])
    extract(p, "comanda", Order)
    schema = p.requests[0].json_schema
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Address"]["additionalProperties"] is False


def test_repairs_invalid_value_with_error_feedback():
    bad = dict(GOOD, quantity=0)
    p = ScriptedProvider([text(json.dumps(bad)), text(json.dumps(GOOD))])
    out = extract(p, "comanda", Order)
    assert out.repairs == 1
    feedback = p.requests[1].messages[-1]["content"]
    assert "quantity" in feedback


def test_repairs_broken_json():
    p = ScriptedProvider([text('{"customer": "Ion",'), text(json.dumps(GOOD))])
    assert extract(p, "x", Order).repairs == 1


def test_custom_validator_triggers_repair():
    p = ScriptedProvider([text(json.dumps(dict(GOOD, customer="  "))), text(json.dumps(GOOD))])
    out = extract(p, "x", Order)
    assert "numele clientului" in p.requests[1].messages[-1]["content"]
    assert out.value.customer == "Ion Rusu"


def test_code_fence_is_tolerated():
    p = ScriptedProvider([text("```json\n" + json.dumps(GOOD) + "\n```")])
    assert extract(p, "x", Order).value.quantity == 3


def test_gives_up_after_max_repairs():
    p = ScriptedProvider(default=text("nu e json"))
    with pytest.raises(StructuredOutputError):
        extract(p, "x", Order, max_repairs=2)
    assert p.calls == 3


def test_usage_is_summed_across_repairs():
    p = ScriptedProvider([text("{", input_tokens=50), text(json.dumps(GOOD), input_tokens=70)])
    assert extract(p, "x", Order).usage.input_tokens == 120


def test_strict_schema_lists_required_fields():
    assert set(strict_schema(Address)["required"]) == {"city", "street"}
