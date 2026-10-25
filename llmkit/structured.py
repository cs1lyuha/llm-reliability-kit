"""Structured outputs: JSON validat cu Pydantic, cu reparare automată.

Pașii:
1. Schema modelului Pydantic pleacă la provider (`json_schema`). Claude o
   respectă prin `output_config.format`; alți provideri o primesc ca
   instrucțiune.
2. Răspunsul trece prin `json.loads` și apoi prin validarea Pydantic, care
   verifică și regulile pe care schema JSON nu le poate exprima (validatori).
3. Dacă validarea pică, trimitem modelului eroarea exactă și îi cerem să
   corecteze. Facem asta de cel mult `max_repairs` ori, apoi renunțăm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from llmkit.providers.base import Provider
from llmkit.types import LLMError, LLMRequest, Usage

M = TypeVar("M", bound=BaseModel)


class StructuredOutputError(LLMError):
    def __init__(self, message: str, last_text: str):
        super().__init__(message)
        self.last_text = last_text


@dataclass
class Extraction(Generic[M]):
    value: M
    repairs: int
    usage: Usage


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Schema JSON a modelului, cu `additionalProperties: false` peste tot
    (cerință pentru structured outputs stricte)."""
    schema = model.model_json_schema()

    def close(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
                node.setdefault("required", list(node.get("properties", {})))
            for value in node.values():
                close(value)
        elif isinstance(node, list):
            for value in node:
                close(value)

    close(schema)
    return schema


def _parse(text: str, model: type[M]) -> M:
    cleaned = text.strip()
    if cleaned.startswith("```"):  # unii provideri pun JSON-ul în code fence
        cleaned = cleaned.strip("`")
        cleaned = cleaned[cleaned.find("{"):]
    return model.model_validate(json.loads(cleaned))


def extract(provider: Provider, prompt: str, model: type[M], *, system: str = "",
            max_repairs: int = 2) -> Extraction[M]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    schema = strict_schema(model)
    usage = Usage()
    last_text = ""
    for repair in range(max_repairs + 1):
        response = provider.complete(LLMRequest(messages=list(messages), system=system, json_schema=schema))
        usage = usage + response.usage
        last_text = response.text
        try:
            return Extraction(_parse(response.text, model), repair, usage)
        except json.JSONDecodeError as err:
            problem = f"Răspunsul nu e JSON valid: {err.msg} (poziția {err.pos})."
        except ValidationError as err:
            details = "; ".join(f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in err.errors())
            problem = f"JSON-ul nu respectă schema: {details}."
        messages += [
            {"role": "assistant", "content": response.text},
            {"role": "user", "content": f"{problem} Întoarce din nou doar obiectul JSON corectat."},
        ]
    raise StructuredOutputError(f"output invalid după {max_repairs} reparări", last_text)
