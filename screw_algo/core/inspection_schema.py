"""VLM 精查 JSON Schema（课件：结构化输出校验）。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from screw_algo.defect_types import DEFECT_CLASS_IDS, normalize_defect_ids


class VlmInspectionSchema(BaseModel):
    violation_detected: bool
    violation_types: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    risk_description: str = ""

    @field_validator("violation_types")
    @classmethod
    def _types_valid(cls, v: list[str]) -> list[str]:
        out = normalize_defect_ids(v)
        if not out:
            return ["good"]
        return out

    def to_inspection_dict(self) -> dict[str, Any]:
        types = self.violation_types
        if "good" in types and len(types) > 1:
            types = [t for t in types if t != "good"]
        if not types:
            types = ["good"]
        defect = self.violation_detected
        if types == ["good"]:
            defect = False
        elif any(t != "good" for t in types):
            defect = True
        for t in types:
            if t not in DEFECT_CLASS_IDS:
                raise ValueError(f"invalid class id: {t}")
        return {
            "violation_detected": defect,
            "violation_types": types,
            "confidence": float(self.confidence),
            "risk_description": self.risk_description,
        }


def validate_inspection_json(data: dict) -> dict[str, Any]:
    parsed = VlmInspectionSchema.model_validate(data)
    return parsed.to_inspection_dict()
