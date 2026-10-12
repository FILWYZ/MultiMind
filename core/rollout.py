                      

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from core.tenant import normalize_tenant_id
from evaluation.quality_gate import evaluate_quality_gate


@dataclass(frozen=True)
class RolloutDecision:
    tenant_id: str
    variant: str
    model: str
    bucket: int


class RolloutManager:
    def __init__(
        self,
        stable_model: str,
        canary_model: str | None = None,
        canary_percent: int | None = None,
        salt: str | None = None,
    ) -> None:
        self.stable_model = stable_model
        self.canary_model = (canary_model or "").strip() or None
        raw_percent = os.getenv("ECHOMIND_CANARY_PERCENT", "0") if canary_percent is None else canary_percent
        try:
            self.canary_percent = max(0, min(100, int(raw_percent)))
        except (TypeError, ValueError):
            self.canary_percent = 0
        self.salt = salt or os.getenv("ECHOMIND_ROLLOUT_SALT", "echomind-stage4")
        self._quality_gate_passed = False
        self._gate_result: dict | None = None

    def select(self, tenant_id: str) -> RolloutDecision:
        tenant_id = normalize_tenant_id(tenant_id)
        digest = hashlib.sha256(f"{self.salt}:{tenant_id}".encode()).hexdigest()
        bucket = int(digest[:8], 16) % 100
        canary = bool(self._quality_gate_passed and self.canary_model and bucket < self.canary_percent)
        return RolloutDecision(
            tenant_id=tenant_id,
            variant="canary" if canary else "stable",
            model=self.canary_model if canary else self.stable_model,
            bucket=bucket,
        )

    def summary(self) -> dict:
        return {
            "stable_model": self.stable_model,
            "canary_model": self.canary_model,
            "canary_percent": self.canary_percent,
            "salt_configured": bool(self.salt),
            "quality_gate_passed": self._quality_gate_passed,
            "last_gate": self._gate_result,
        }

    def promote_candidate(self, model: str, percent: int, metrics: dict, baseline: dict,
                          candidate_dataset_fingerprint: str = "",
                          baseline_dataset_fingerprint: str = "") -> dict:
        candidate_model = str(model or "").strip()
        if not candidate_model:
            raise ValueError("candidate model is required")
        gate = evaluate_quality_gate(metrics, baseline=baseline).to_dict()
        dataset_match = bool(candidate_dataset_fingerprint) and candidate_dataset_fingerprint == baseline_dataset_fingerprint
        gate["checks"]["dataset_fingerprint_match"] = dataset_match
        if not dataset_match:
            gate["blockers"].append("dataset_fingerprint_match")
        gate["passed"] = gate["passed"] and dataset_match
        self._gate_result = gate
        self._quality_gate_passed = bool(gate["passed"])
        if not self._quality_gate_passed:
            self.canary_percent = 0
            return {"promoted": False, "gate": gate, "canary_percent": 0}
        self.canary_model = candidate_model
        self.canary_percent = max(1, min(100, int(percent)))
        return {"promoted": True, "gate": gate, "canary_percent": self.canary_percent,
                "canary_model": self.canary_model}

    def assess_online_quality(self, summary: dict, *, min_samples: int = 30,
                              min_rating: float = 3.8) -> dict:
        canary = (summary.get("by_variant") or {}).get("canary", {})
        count = int(canary.get("count", 0) or 0)
        rating = canary.get("mean_rating")
        if count < min_samples or rating is None:
            return {"observed": False, "passed": None, "sample_count": count,
                    "required_samples": min_samples, "mean_rating": rating,
                    "action": "keep_small_canary_and_collect_feedback"}
        passed = float(rating) >= min_rating
        if not passed:
            self.canary_percent = 0
            self._quality_gate_passed = False
        return {"observed": True, "passed": passed, "sample_count": count,
                "required_samples": min_samples, "mean_rating": rating,
                "action": "continue_canary" if passed else "rollback_to_stable"}
